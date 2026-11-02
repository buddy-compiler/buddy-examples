from math import prod

import torch
from buddy.compiler.graph.operation import (
    AddOp,
    BatchMatmulOp,
    CloneOp,
    ExpandOp,
    PermuteOp,
    SubOp,
    UnsqueezeOp,
    ViewOp,
)
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from compiler.frontend.quant import FP32ToMXINT8QuantOp, MXINT8ToFP32DequantOp


def quantize_keys_in_cache_layout(graph):
    """Keep two-term key quantization in the cache's layout, then transpose.

    Layout changes preserve each quantization group. Moving the whole
    head/residual/tail expression keeps FP32 subtraction and addition in
    their original order and avoids transposing packed bytes on the CPU.
    """

    def redirect(user, index, source):
        previous = graph.node_table[user.args[index]]
        previous.children.remove(user.name)
        user.parents[user.parents.index(previous.name)] = source.name
        user.args[index] = source.name
        source.add_children(user.name)

    for product in graph.body:
        if not isinstance(product, BatchMatmulOp):
            continue
        total = graph.node_table[product.args[1]]
        if not isinstance(total, AddOp):
            continue
        head, tail = [graph.node_table[name] for name in total.args]
        if not all(isinstance(node, MXINT8ToFP32DequantOp) for node in (head, tail)):
            continue
        hq, tq = [graph.node_table[node.args[0]] for node in (head, tail)]
        if not all(isinstance(node, FP32ToMXINT8QuantOp) for node in (hq, tq)):
            continue
        residual = graph.node_table[tq.args[0]]
        source = graph.node_table[hq.args[0]]
        if (
            not isinstance(residual, SubOp)
            or residual.args != [source.name, head.name]
            or set(source.children) != {hq.name, residual.name}
        ):
            continue
        axis = hq.args[1] % len(source.tensor_meta["shape"])
        base, views = source, []
        while isinstance(base, (ViewOp, PermuteOp, ExpandOp, UnsqueezeOp, CloneOp)):
            if views and base.children != [views[-1].name]:
                break
            parent = graph.node_table[base.args[0]]
            before = [
                d.node.expr if isinstance(d, torch.SymInt) else d
                for d in parent.tensor_meta["shape"]
            ]
            after = [
                d.node.expr if isinstance(d, torch.SymInt) else d
                for d in base.tensor_meta["shape"]
            ]
            if isinstance(base, PermuteOp):
                axis = base.args[1][axis]
            elif isinstance(base, ViewOp):
                axes = [
                    i
                    for i, size in enumerate(before)
                    if size == after[axis]
                    and prod(before[:i]) == prod(after[:axis])
                    and prod(before[i + 1 :]) == prod(after[axis + 1 :])
                ]
                if len(axes) != 1:
                    break
                axis = axes[0]
            elif isinstance(base, ExpandOp):
                mapped = axis - (len(after) - len(before))
                if mapped < 0 or before[mapped] != after[axis]:
                    break
                axis = mapped
            elif isinstance(base, UnsqueezeOp):
                inserted = base.args[1] % len(after)
                if axis == inserted:
                    break
                axis -= axis > inserted
            views.append(base)
            base = parent
        if not any(isinstance(view, PermuteOp) for view in views):
            continue
        shape = list(base.tensor_meta["shape"])
        packed = list(shape)
        packed[axis] += packed[axis] // 32
        for node in (hq, tq):
            node.args[1] = axis
            node.tensor_meta["shape"] = list(packed)
        for node in (head, tail):
            node.args[1] = axis
        for node in (head, tail, residual, total):
            node.tensor_meta["shape"] = list(shape)
        redirect(hq, 0, base)
        redirect(residual, 0, base)
        redirect(views[-1], 0, total)
        redirect(product, 1, source)
    sort_graph(graph)


def group_queries(graph):
    """Use query heads sharing a KV head as matrix rows, without repeating KV.

    [Hq, S, K] @ repeat([Hkv, K, N]) becomes
    [Hkv, (Hq/Hkv)*S, K] @ [Hkv, K, N]. The existing result view restores
    query-head order. Quantization still groups the same K values per row.
    """
    for product in list(graph.body):
        if not isinstance(product, BatchMatmulOp):
            continue
        left, right = [graph.node_table[name] for name in product.args]
        assert isinstance(left, ViewOp) and isinstance(right, ViewOp)
        expanded = graph.node_table[right.args[0]]
        assert isinstance(expanded, ExpandOp)
        repeated = graph.node_table[expanded.args[0]]
        transpose = repeated if isinstance(repeated, PermuteOp) else None
        if transpose:
            repeated = graph.node_table[transpose.args[0]]
        assert isinstance(repeated, ViewOp)
        clone = graph.node_table[repeated.args[0]]
        assert isinstance(clone, CloneOp)
        broadcast = graph.node_table[clone.args[0]]
        assert isinstance(broadcast, ExpandOp)
        unsqueeze = graph.node_table[broadcast.args[0]]
        assert isinstance(unsqueeze, UnsqueezeOp) and unsqueeze.args[1] == 2
        cache = graph.node_table[unsqueeze.args[0]]
        heads = cache.tensor_meta["shape"][1]
        groups = broadcast.tensor_meta["shape"][2]
        assert repeated.tensor_meta["shape"][1] == heads * groups
        assert all(
            isinstance(graph.node_table[name], ViewOp) for name in product.children
        )

        # Remove the repeat, retaining the cache's logical four-dimensional view.
        repeated.args[0] = cache.name
        repeated.parents.remove(clone.name)
        clone.children.remove(repeated.name)
        repeated.add_parent(cache.name)
        cache.add_children(repeated.name)
        for node in [repeated, *([transpose] if transpose else []), expanded]:
            shape = list(node.tensor_meta["shape"])
            shape[1] = heads
            node.tensor_meta["shape"] = shape
            if isinstance(node, (ViewOp, ExpandOp)):
                node.args[1][1] = heads
        for node in (clone, broadcast, unsqueeze):
            assert not node.children
            graph.delete_node(node, [graph.node_table[name] for name in node.parents])

        # Both operands' outer expands only prepare torch.matmul's batch shape.
        for view in (left, right):
            expand = graph.node_table[view.args[0]]
            assert isinstance(expand, ExpandOp)
            source = graph.node_table[expand.args[0]]
            view.args[0] = source.name
            view.parents.remove(expand.name)
            expand.children.remove(view.name)
            view.add_parent(source.name)
            source.add_children(view.name)
            if not expand.children:
                graph.delete_node(
                    expand, [graph.node_table[name] for name in expand.parents]
                )
            view.args[1][0] = heads
            shape = list(view.tensor_meta["shape"])
            shape[0] = heads
            if view is left:
                view.args[1][1] = -1
                shape[1] *= groups
            view.tensor_meta["shape"] = shape
        shape = list(product.tensor_meta["shape"])
        shape[0] = heads
        shape[1] *= groups
        product.tensor_meta["shape"] = shape

from math import prod

import torch
from buddy.compiler.graph.operation import (
    AddOp,
    BatchMatmulOp,
    CloneOp,
    ExpandOp,
    PermuteOp,
    OpType,
    SubOp,
    UnsqueezeOp,
    ViewOp,
)
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from compiler.frontend.quant import FP32ToMXINT8QuantOp, MXINT8ToFP32DequantOp


def quantize_kv_in_cache_layout(graph):
    """Quantize keys and values before broadcasting or transposing the cache.

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
        residual = None
        if isinstance(total, AddOp):
            head, tail = [graph.node_table[name] for name in total.args]
            if not all(
                isinstance(node, MXINT8ToFP32DequantOp) for node in (head, tail)
            ):
                continue
            hq, tq = [graph.node_table[node.args[0]] for node in (head, tail)]
            if not all(isinstance(node, FP32ToMXINT8QuantOp) for node in (hq, tq)):
                continue
            residual = graph.node_table[tq.args[0]]
            if not isinstance(residual, SubOp) or residual.args != [
                hq.args[0],
                head.name,
            ]:
                continue
            quantized, restored = (hq, tq), (head, tail, residual, total)
        elif isinstance(total, MXINT8ToFP32DequantOp):
            hq = graph.node_table[total.args[0]]
            if not isinstance(hq, FP32ToMXINT8QuantOp):
                continue
            quantized, restored = (hq,), (total,)
        else:
            continue
        source = graph.node_table[hq.args[0]]
        users = {hq.name, residual.name} if residual else {hq.name}
        if set(source.children) != users:
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
        if not any(isinstance(view, (PermuteOp, ExpandOp)) for view in views):
            continue
        shape = list(base.tensor_meta["shape"])
        packed = list(shape)
        packed[axis] += packed[axis] // 32
        for node in quantized:
            node.args[1] = axis
            node.tensor_meta["shape"] = list(packed)
        for node in restored:
            if isinstance(node, MXINT8ToFP32DequantOp):
                node.args[1] = axis
            node.tensor_meta["shape"] = list(shape)
        redirect(hq, 0, base)
        if residual:
            redirect(residual, 0, base)
        redirect(views[-1], 0, total)
        redirect(product, 1, source)
    sort_graph(graph)


def reuse_cache_quantization(graph):
    from frontend.quant.graph import replace_argument

    values = {}
    reused = 0
    for node in list(graph.body):
        if not isinstance(node, (FP32ToMXINT8QuantOp, MXINT8ToFP32DequantOp)):
            if not isinstance(node, SubOp) or not isinstance(
                graph.node_table.get(node.args[1]), MXINT8ToFP32DequantOp
            ):
                continue
        key = (type(node), tuple(node.args), repr(node.tensor_meta))
        if key not in values:
            values[key] = node
            continue
        for name in list(node.children):
            user = graph.node_table[name]
            for index, argument in enumerate(list(user.args)):
                if argument == node.name:
                    replace_argument(graph, user, index, values[key])
        graph.delete_node(node, [graph.node_table[name] for name in node.parents])
        reused += 1
    print(f"Reused {reused} identical KV quantization operations")
    sort_graph(graph)


def split_attention_products(graph):
    """Use two MXINT8 products for a quantized operand and its residual.

    This changes rounding from Q @ (Kh + Kt) to Q @ Kh + Q @ Kt and is
    validated by the model recipe's perplexity calibration.
    """
    for product in list(graph.body):
        if not isinstance(product, BatchMatmulOp):
            continue
        for operand_index in (0, 1):
            source = graph.node_table[product.args[operand_index]]
            views = []
            while isinstance(
                source, (ViewOp, PermuteOp, ExpandOp, UnsqueezeOp, CloneOp)
            ):
                views.append(source)
                source = graph.node_table[source.args[0]]
            if not isinstance(source, AddOp):
                continue
            terms = [graph.node_table[name] for name in source.args]
            if all(isinstance(term, MXINT8ToFP32DequantOp) for term in terms):
                break
        else:
            continue
        products = []
        for suffix, term in zip(("head", "tail"), terms, strict=True):
            value = term
            for view in reversed(views):
                clone = type(view)()
                clone.name = f"{product.name}_{suffix}_{view.name}"
                clone.tensor_meta = dict(view.tensor_meta)
                for argument in [value.name, *view.args[1:]]:
                    clone.add_argument(argument)
                for parent in view.parents:
                    name = value.name if parent == view.args[0] else parent
                    clone.add_parent(name)
                    graph.node_table[name].add_children(clone.name)
                graph.add_node(clone)
                value = clone
            result = BatchMatmulOp()
            result.name = f"{product.name}_{suffix}_product"
            result.tensor_meta = dict(product.tensor_meta)
            for index, original in enumerate(product.args):
                argument = value.name if index == operand_index else original
                result.add_argument(argument)
                result.add_parent(argument)
                graph.node_table[argument].add_children(result.name)
            graph.add_node(result)
            products.append(result)
        combined = AddOp()
        combined.name = product.name
        graph.displace_node(product, combined)
        combined._op_type = OpType.BroadcastType
        for index, result in enumerate(products):
            previous = graph.node_table[combined.args[index]]
            previous.children.remove(combined.name)
            combined.parents[combined.parents.index(previous.name)] = result.name
            combined.args[index] = result.name
            result.add_children(combined.name)
        for unused in [*views, source]:
            if not unused.children:
                graph.delete_node(
                    unused, [graph.node_table[name] for name in unused.parents]
                )
    sort_graph(graph)

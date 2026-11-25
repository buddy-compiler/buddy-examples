from math import prod

import buddy_mlir.ir as ir
import torch
from buddy_mlir.dialects import arith, linalg, tensor
from buddy.compiler.graph.operation import (
    AddOp,
    AmaxOp,
    AnyOp,
    BatchMatmulOp,
    CatOp,
    CloneOp,
    CosOp,
    ExpandOp,
    EqualOp,
    LogicalNotOp,
    MulOp,
    NegOp,
    Op,
    PermuteOp,
    OpType,
    SliceOp,
    SinOp,
    SubOp,
    UnsqueezeOp,
    ViewOp,
)
from buddy.compiler.graph.sizes import broadcast_sizes, mlir_shape
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from compiler.frontend.quant import FP32ToMXINT8QuantOp, MXINT8ToFP32DequantOp


def reuse_rotary_views(graph):
    from frontend.quant.graph import replace_argument

    views = {}
    reused = 0
    for node in list(graph.body):
        if not isinstance(node, UnsqueezeOp) or node.args[1] != 1:
            continue
        source = graph.node_table[node.args[0]]
        if not isinstance(source, MulOp) or source.args[1] != 1.0:
            continue
        if not isinstance(graph.node_table[source.args[0]], (CosOp, SinOp)):
            continue
        key = (source.name, repr(node.tensor_meta))
        if key not in views:
            views[key] = node
            continue
        for name in list(node.children):
            user = graph.node_table[name]
            for index, argument in enumerate(list(user.args)):
                if argument == node.name:
                    replace_argument(graph, user, index, views[key])
        if node.children:
            raise ValueError(f"{node.name}: expected direct rotary-view users")
        graph.delete_node(node, [source])
        reused += 1
    print(f"Reused {reused} identical rotary-table views")


def reuse_softmax_maximum(graph):
    # A NaN-propagating maximum is -inf exactly when every entry is -inf.
    # Reuse softmax's maximum instead of scanning the score matrix again.
    for node in list(graph.body):
        if not isinstance(node, LogicalNotOp):
            continue
        reduction = graph.node_table[node.args[0]]
        if not isinstance(reduction, AnyOp) or reduction.args[1:] != [-1, True]:
            continue
        inverted = graph.node_table[reduction.args[0]]
        if not isinstance(inverted, LogicalNotOp):
            continue
        comparison = graph.node_table[inverted.args[0]]
        if not isinstance(comparison, EqualOp) or comparison.args[1] != -float("inf"):
            continue
        (maximum,) = [
            candidate
            for candidate in graph.body
            if isinstance(candidate, AmaxOp)
            and candidate.args == [comparison.args[0], [-1], True]
        ]
        replacement = EqualOp()
        replacement.name = node.name
        graph.displace_node(node, replacement)
        reduction.children.remove(replacement.name)
        replacement.args[:] = [maximum.name, -float("inf")]
        replacement.parents[:] = [maximum.name]
        maximum.add_children(replacement.name)
        for unused in (reduction, inverted, comparison):
            if not unused.children:
                graph.delete_node(
                    unused, [graph.node_table[name] for name in unused.parents]
                )
    sort_graph(graph)


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


class RotaryOp(Op):
    """RoPE of the heads of q or k: RotaryOp(x, cos, sin) with x [1, T, H, D]
    token after token, as its Linear writes it, cos and sin [1, 1, T, D];
    the result is [1, H, T, D], head after head:
    x' * cos + rotate_half(x') * sin, x' = x.permute(0, 2, 1, 3),
    rotate_half(x') = cat(-x'[..., D/2:], x'[..., :D/2])."""

    def __init__(self):
        super().__init__()
        self._op_type = OpType.Unfusable


def rotary_op(node, symbol_table):
    """One linalg.generic: it reads x through the head permutation and the
    rotated half through (d + D/2) mod D, negated in the first half, so
    neither the permuted x nor rotate_half is materialized. The same
    multiplications and addition per element as the graph it replaces."""
    x, cos, sin = [symbol_table[(name, 0)] for name in node.args]
    f32 = ir.F32Type.get()
    result = ir.RankedTensorType.get(mlir_shape(node.tensor_meta["shape"]), f32)
    width = node.tensor_meta["shape"][3]
    b, h, t, d = [ir.AffineExpr.get_dim(i) for i in range(4)]
    swapped = ir.AffineExpr.get_mod(
        ir.AffineExpr.get_add(d, ir.AffineExpr.get_constant(width // 2)),
        ir.AffineExpr.get_constant(width),
    )
    table = [
        ir.AffineExpr.get_constant(0) if extent == 1 else dim
        for extent, dim in zip(ir.RankedTensorType(cos.type).shape, (b, h, t, d))
    ]
    maps = [
        ir.AffineMap.get(4, 0, [b, t, h, d]),
        ir.AffineMap.get(4, 0, [b, t, h, swapped]),
        ir.AffineMap.get(4, 0, table),
        ir.AffineMap.get(4, 0, table),
        ir.AffineMap.get_identity(4),
    ]
    empty = tensor.EmptyOp(broadcast_sizes(result, [cos]), f32)
    rope = linalg.GenericOp(
        [result],
        [x, x, cos, sin],
        [empty.result],
        ir.ArrayAttr.get([ir.AffineMapAttr.get(m) for m in maps]),
        ir.ArrayAttr.get([ir.Attribute.parse("#linalg.iterator_type<parallel>")] * 4),
    )
    block = ir.Block.create_at_start(rope.region, [f32] * 5)
    value, other, c, s, _ = block.arguments
    with ir.InsertionPoint(block):
        index = ir.IndexType.get()
        first = arith.CmpIOp(
            arith.CmpIPredicate.ult,
            linalg.IndexOp(
                ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 3)
            ).result,
            arith.ConstantOp(index, width // 2).result,
        ).result
        rotated = arith.SelectOp(first, arith.NegFOp(other).result, other).result
        linalg.YieldOp(
            [
                arith.AddFOp(
                    arith.MulFOp(value, c).result, arith.MulFOp(rotated, s).result
                ).result
            ]
        )
    return rope


ops_registry = {"RotaryOp": rotary_op}


def fuse_rotary(device):
    """RotaryOp for the RoPE of every attention graph of the poly harts:
    add(mul(x', cos), mul(cat(neg(x'[..., D/2:]), x'[..., :D/2]), sin)) with
    x' = permute(x, [0, 2, 1, 3]) becomes RotaryOp(x, cos, sin) under the
    name of the add, so the CPU of a poly hart neither transposes q and k nor
    concatenates their rotated halves before the launches that read them."""
    for segment in device.graphs.values():
        if segment.graph_class not in ("attention_global", "attention_local"):
            continue
        nodes = {node.name: node for _, node in segment.ops}
        users = {}
        for _, node in segment.ops:
            for name in node.parents:
                users.setdefault(name, set()).add(node.name)
        replaced, removed = {}, set()
        for _, add in segment.ops:
            if not isinstance(add, AddOp) or not all(
                isinstance(nodes.get(name), MulOp) for name in add.args
            ):
                continue
            products = [nodes[name] for name in add.args]
            rotated = [
                (product, nodes.get(product.args[0]))
                for product in products
                if isinstance(nodes.get(product.args[0]), CatOp)
            ]
            if len(rotated) != 1:
                continue
            ((second, cat),) = rotated
            (first,) = [product for product in products if product is not second]
            permuted = nodes.get(first.args[0])
            if not isinstance(permuted, PermuteOp) or list(permuted.args[1]) != [
                0,
                2,
                1,
                3,
            ]:
                continue
            width = permuted.tensor_meta["shape"][3]
            (negated, low), axis = cat.args
            negated, low = nodes[negated], nodes[low]
            if not isinstance(negated, NegOp):
                continue
            high = nodes[negated.args[0]]
            if not (
                axis % 4 == 3
                and isinstance(low, SliceOp)
                and isinstance(high, SliceOp)
                and low.args[:4] == [permuted.name, 3, 0, width // 2]
                and high.args[:3] == [permuted.name, 3, width // 2]
                and users[permuted.name] == {first.name, low.name, high.name}
            ):
                continue
            pattern = [permuted, first, second, cat, negated, low, high]
            if any(
                users[node.name] - {n.name for n in pattern} - {add.name}
                or node.name in segment.outputs
                for node in pattern
            ):
                continue
            rope = RotaryOp()
            rope.name = add.name
            for name in (permuted.args[0], first.args[1], second.args[1]):
                rope.add_argument(name)
                rope.add_parent(name)
            rope.tensor_meta = dict(add.tensor_meta)
            replaced[add.name] = rope
            removed |= {node.name for node in pattern}
        if not replaced:
            raise ValueError(f"{segment.graph}: no RoPE")
        segment.ops = [
            (key, replaced.get(node.name, node))
            for key, node in segment.ops
            if node.name not in removed
        ]

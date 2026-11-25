"""Partition a graph across harts with tensor parallelism, as a device
program.

Hart 0 is the reduce hart; harts 1..count run ranks 0..count-1. The recipe
chooses the Linears (TransposeMatmulFusedOp) split along their output
channels: their MXINT8 blocks (or units of them the recipe names: the
heads that share a key head) are dealt out to the ranks, the first ranks one
more when they do not divide evenly, and a fused Linear (q, k, v or gate, up
in one) deals each of its segments out on its own. Every value computed from
a sharded value is sharded along one axis, and each rank computes its pieces
of it. A Linear whose input is sharded is split along K: each rank computes a
partial sum, and the reduce hart adds them up. A top-k along the sharded
axis (an argmax is k = 1) is split the same way: each rank takes the k
largest values of its piece and their indices there, the first index first
among equal values; the reduce hart joins them in rank order, each index
offset by the start of the rank's piece. The joined candidates hold the k
largest of the whole axis, and the first largest candidate in order is the
first maximum of the axis. The other Linears run whole.
States (the KV cache) are sharded inputs, dealt out the same way: each rank
owns its slice. Sharded
outputs are joined by the reduce hart. Weights are packed in complete 32-value blocks along K; a split keeps
whole blocks.

The other operations run on the reduce hart, except the `replicated` ones
(positions, masks, rotary tables, and the keys and values when there are
fewer KV heads than ranks): every segment that uses them computes them
itself, and reads its pieces of those that meet a sharded value along its
sharded axis: the reduce hart deals its pieces of a value it computes out to
the ranks; a parameter is sliced offline; a value the same at every offset of
the axis (a mask broadcast over the heads) is read from the start by every
rank alike.

The result is a device program (Device): each hart's operations cut into
segments where they read a value another hart computes, and the outputs the
reduce hart computes (the joins of the sharded ones, the replicated ones)
apart. The recipe names the semantic graph class of every segment (the
attention of a rank, its MLP, the end of a block, ...): the segments of a
class are instances of one graph, defined once with its parameters as
arguments and launched on the class of its hart (the recipe names the class
of every hart). Which hart
runs a launch and how a value moves between harts are the device
compiler's.
"""

import copy
import math

import buddy_mlir.ir as ir
import numpy
import sympy
import torch

from buddy.compiler.graph import Graph, NodeType, OpType
from buddy.compiler.graph.operation import (
    AddMMOp,
    AddOp,
    AmaxOp,
    BatchMatmulOp,
    CatOp,
    CloneOp,
    ExpandOp,
    FullOp,
    GetItemOp,
    MatmulOp,
    MeanOp,
    OutputOp,
    PermuteOp,
    PlaceholderOp,
    SelectOp,
    SliceOp,
    SplitOp,
    SqueezeDimOp,
    SumDimOp,
    SymSizeOp,
    TopkOp,
    TransposeMatmulFusedOp,
    UnbindOp,
    UnsqueezeOp,
    ViewOp,
)
from buddy.compiler.ops.utils import mlir_element_type_get
from buddy.compiler.graph.type import TensorDType

from .quant import (
    FP32ToMXINT8QuantOp,
    MXINT8ActivationOp,
    MXINT8ToFP32DequantOp,
    MXINT5ToFP32DequantOp,
)
from .mxfp4 import MXFP4ToFP32DequantOp


def size(dimension):
    """A dimension as a sympy expression, symbolic or not."""
    if isinstance(dimension, torch.SymInt):
        return dimension.node.expr
    return sympy.Integer(dimension)


def total(pieces):
    return sum(end - start for start, end in pieces)


def follow(graph, node, axes, pieces):
    """The sharded axis of the result of `node` and the pieces each rank
    holds along it, from those of its first sharded argument."""
    parent = next(graph.node_table[name] for name in node.parents if name in axes)
    axis, held = axes[parent.name], pieces[parent.name]
    if isinstance(node, GetItemOp):
        source = graph.node_table[parent.args[0]]
        dimension = parent.args[1 if isinstance(parent, UnbindOp) else 2] % len(
            source.tensor_meta["shape"]
        )
        if isinstance(parent, UnbindOp):
            return axis - (dimension < axis), held
        if dimension != axis:
            return axis, held
        # A chunk of a split along the sharded axis: each rank its pieces in it.
        width = parent.args[1]
        low, high = node.args[1] * width, (node.args[1] + 1) * width
        return axis, [
            [
                (max(s, low) - low, min(e, high) - low)
                for s, e in rank
                if s < high and e > low
            ]
            for rank in held
        ]
    rank = len(parent.tensor_meta["shape"])
    if (
        isinstance(node, (UnbindOp, SqueezeDimOp, SelectOp))
        and node.args[1] % rank == axis
    ):
        raise ValueError(f"{node.name}: drops the sharded axis")
    if isinstance(node, (UnbindOp, SplitOp)):
        return axis, held
    if isinstance(node, (SqueezeDimOp, SelectOp)):
        return axis - (node.args[1] % rank < axis), held
    shape = node.tensor_meta["shape"]
    if isinstance(node, PermuteOp):
        return list(node.args[1]).index(axis), held
    if isinstance(node, UnsqueezeOp):
        return axis + (node.args[1] % len(shape) <= axis), held
    if isinstance(node, ViewOp):
        # The axis behind as many values as the axes before the sharded one.
        source = parent.tensor_meta["shape"]
        prefix = sympy.Mul(*[size(d) for d in source[:axis]])
        candidates = [
            index
            for index in range(len(shape))
            if shape[index] != 1
            and sympy.simplify(sympy.Mul(*[size(d) for d in shape[:index]]) - prefix)
            == 0
        ]
        if len(candidates) != 1:
            raise ValueError(f"{node.name}: no sharded axis in {list(shape)}")
        (result,) = candidates
        inner = sympy.Mul(*[size(d) for d in source[axis + 1 :]])
        if len(held[0]) > 1 and shape[result] == len(held[0]):
            # The segments of a fused Linear become their own axis; each
            # rank holds the same piece of every segment.
            segment = source[axis] // len(held[0])
            for rank in held:
                if any(
                    piece != (s - i * segment, e - i * segment)
                    for i, (s, e) in enumerate(rank)
                    for piece in [rank[0]]
                ):
                    raise ValueError(f"{node.name}: uneven segments")
            factor = sympy.cancel(
                inner / sympy.Mul(*[size(d) for d in shape[result + 2 :]])
            )
            held = [[rank[0]] for rank in held]
            result += 1
        else:
            factor = sympy.cancel(
                inner / sympy.Mul(*[size(d) for d in shape[result + 1 :]])
            )
        scaled = [[(s * factor, e * factor) for s, e in rank] for rank in held]
        if not all(
            sympy.sympify(v).is_integer
            for rank in scaled
            for piece in rank
            for v in piece
        ):
            raise ValueError(f"{node.name}: the pieces do not fit {list(shape)}")
        return result, [[(int(s), int(e)) for s, e in rank] for rank in scaled]
    if isinstance(node, BatchMatmulOp):
        # The batches, or the rows of the first operand (the query heads of
        # one key head, whose keys every rank reads whole).
        # Keys every rank computed whole itself hold the same piece on each.
        other = pieces.get(node.args[1])
        if (
            axis == 1
            and node.args[0] == parent.name
            and (other is None or all(rank == other[0] for rank in other))
        ):
            return 1, held
        if axis != 0:
            raise ValueError(f"{node.name}: sharded along {axis}, not the batch")
        return 0, held
    if isinstance(node, (SliceOp, CatOp)) and node.args[1] % rank == axis:
        # What every rank holds whole, every rank slices and joins whole.
        if any(h != [(0, parent.tensor_meta["shape"][axis])] for h in held):
            raise ValueError(f"{node.name}: slices or joins the sharded axis")
        return axis, [[(0, shape[axis])] for _ in held]
    if isinstance(node, (MeanOp, SumDimOp, AmaxOp)):
        reduced = [d % rank for d in node.args[1]]
        if axis in reduced:
            raise ValueError(f"{node.name}: reduces the sharded axis")
        keep = len(node.args) > 2 and node.args[2]
        return axis - (0 if keep else sum(d < axis for d in reduced)), held
    # Elementwise: the arguments broadcast from the right.
    return axis + len(shape) - rank, held


def separate_biases(graph):
    """Each Linear with a bias (AddMMOp over a restored weight) as its
    product and the addition of the bias, so that the product splits like
    any Linear: along its output channels, each rank its slice of the bias;
    along K, the reduce hart adding the bias to the partial sums."""
    for node in list(graph.body):
        if not isinstance(node, AddMMOp):
            continue
        bias, activation, weight = node.args
        product = MatmulOp()
        product.name = f"{node.name}_product"
        for name in (activation, weight):
            product.add_argument(name)
            product.add_parent(name)
            parent = graph.node_table[name]
            parent._children[parent._children.index(node.name)] = product.name
        product.tensor_meta = dict(node.tensor_meta)
        if not restoring(graph, product):
            for name in (activation, weight):
                parent = graph.node_table[name]
                parent._children[parent._children.index(product.name)] = node.name
            continue
        added = AddOp()
        added.name = node.name
        for name in (product.name, bias):
            added.add_argument(name)
            added.add_parent(name)
        added.tensor_meta = dict(node.tensor_meta)
        product.add_children(added.name)
        for child in node.children:
            added.add_children(child)
        index = graph.body.index(node)
        graph._body = graph.body[:index] + [product, added] + graph.body[index + 1 :]
        graph.node_table[product.name] = product
        graph.node_table[added.name] = added


def restoring(graph, node):
    """The restored weight of a Linear and the operand the product reads: a
    fused transposed product reads the restored weight, a product the
    transpose of it; None for another operation."""
    if isinstance(node, TransposeMatmulFusedOp):
        restored = graph.node_table[node.args[1]]
        return restored, restored
    if isinstance(node, MatmulOp):
        operand = graph.node_table[node.args[1]]
        if isinstance(operand, PermuteOp) and list(operand.args[1]) == [1, 0]:
            restored = graph.node_table[operand.args[0]]
            if isinstance(
                restored, (MXINT8ToFP32DequantOp, MXFP4ToFP32DequantOp)
            ) and isinstance(graph.node_table[restored.args[0]], PlaceholderOp):
                return restored, operand
    return None


def spanned(graph, node, axes):
    """The unsharded arguments of the sharded `node` that span its sharded
    axis whole, with their axis along it: each rank reads its pieces."""
    shape = node.tensor_meta["shape"]
    result = {}
    # The second operand of a product sharded along its rows is read whole.
    if isinstance(node, BatchMatmulOp) and axes[node.name] == 1:
        return result
    for name in node.parents:
        if name in axes or (
            isinstance(node, MXINT8ActivationOp) and name == node.args[2]
        ):
            continue
        # Scalars (symbolic sizes) have no shape.
        other = graph.node_table[name].tensor_meta.get("shape", [])
        offset = axes[node.name] - len(shape) + len(other)
        if 0 <= offset and other[offset] != 1:
            result[name] = offset
    return result


def uniform(graph, name, axis):
    """Whether the value `name` is the same at every offset along `axis`:
    filled, broadcast along it, permuted or reshaped keeping it, quantized
    along another axis, or computed elementwise from such values."""
    node = graph.node_table[name]
    shape = node.tensor_meta["shape"]
    if isinstance(node, FullOp):
        return True
    if isinstance(node, ExpandOp):
        before = graph.node_table[node.args[0]].tensor_meta["shape"]
        inner = axis - len(shape) + len(before)
        return inner < 0 or before[inner] == 1 or uniform(graph, node.args[0], inner)
    if isinstance(node, PermuteOp):
        return uniform(graph, node.args[0], list(node.args[1])[axis])
    if isinstance(node, ViewOp):
        source = graph.node_table[node.args[0]].tensor_meta["shape"]
        prefix = sympy.Mul(*[size(d) for d in shape[:axis]])
        for inner in range(len(source)):
            if (
                sympy.simplify(sympy.Mul(*[size(d) for d in source[:inner]]) - prefix)
                == 0
                and sympy.simplify(size(source[inner]) - size(shape[axis])) == 0
            ):
                return uniform(graph, node.args[0], inner)
        return False
    if isinstance(
        node,
        (
            FP32ToMXINT8QuantOp,
            MXINT8ToFP32DequantOp,
            MXINT5ToFP32DequantOp,
            MXFP4ToFP32DequantOp,
        ),
    ):
        # MXINT8 blocks along another axis.
        return node.args[1] % len(shape) != axis and uniform(graph, node.args[0], axis)
    if node._op_type not in (OpType.ElementwiseType, OpType.BroadcastType):
        return False
    tensors = [
        graph.node_table[p]
        for p in node.parents
        if graph.node_table[p].tensor_meta.get("shape")
    ]
    if not tensors:
        return False
    for parent in tensors:
        other = parent.tensor_meta["shape"]
        inner = axis - len(shape) + len(other)
        if inner >= 0 and other[inner] != 1 and not uniform(graph, parent.name, inner):
            return False
    return True


def copied(node, name, arguments, shape=None):
    """A copy of `node` named `name`, over `arguments`, with result `shape`
    (or that of `node`)."""
    result = copy.copy(node)
    result._name = name
    result._arguments = arguments
    result._keyword_arguments = dict(node.kwargs)
    result._args_index = list(node._args_index)
    names = [a for a in arguments if isinstance(a, str)]
    names += [
        b for a in arguments if isinstance(a, list) for b in a if isinstance(b, str)
    ]
    result._parents = names
    result._children = []
    result._tensor_meta = dict(node.tensor_meta)
    if shape is not None:
        result._tensor_meta["shape"] = list(shape)
    return result


def narrow_broadcast(graph, source, axis, width, suffix):
    """Build only the requested heads of a broadcast through layout operations."""
    chain = []
    node = source
    while isinstance(node, (ExpandOp, ViewOp, PermuteOp, UnsqueezeOp, CloneOp)):
        shape = list(node.tensor_meta["shape"])
        parent = graph.node_table[node.args[0]]
        before = parent.tensor_meta["shape"]
        chain.append((node, axis))
        if isinstance(node, ExpandOp):
            inner = axis - len(shape) + len(before)
            if inner < 0 or before[inner] == 1:
                break
            axis = inner
        elif isinstance(node, PermuteOp):
            axis = node.args[1][axis]
        elif isinstance(node, UnsqueezeOp):
            axis -= node.args[1] % len(shape) < axis
        elif isinstance(node, ViewOp):
            axes = [
                i
                for i, dimension in enumerate(before)
                if size(dimension) == size(shape[axis])
                and sympy.Mul(*map(size, before[:i]))
                == sympy.Mul(*map(size, shape[:axis]))
                and sympy.Mul(*map(size, before[i + 1 :]))
                == sympy.Mul(*map(size, shape[axis + 1 :]))
            ]
            if len(axes) != 1:
                return []
            (axis,) = axes
        node = parent
    else:
        return []
    result = []
    name = parent.name
    for node, axis in reversed(chain):
        shape = list(node.tensor_meta["shape"])
        shape[axis] = width
        arguments = list(node.args)
        arguments[0] = name
        if isinstance(node, (ViewOp, ExpandOp)):
            arguments[1] = list(arguments[1])
            arguments[1][axis] = width
        node = copied(node, node.name + suffix, arguments, shape)
        result.append(node)
        name = node.name
    return result


def partition(
    graph,
    params,
    count,
    states,
    replicated,
    columns,
    segments,
    per_rank,
    shared,
    native,
    classify,
    quantized_partials,
    units,
    classes,
    sink,
):
    """The device program of the reduce hart and the `count` ranks.
    states: the sharded inputs and their sharded axis. replicated: the
    operations every hart computes for itself. columns: the
    Linears split along their output channels. segments: the fused Linears
    among them (q, k, v or gate, up in one) and their number of segments,
    each split across the ranks on its own. per_rank: replicated preparation
    regions materialized once per consuming hart. shared: replicated intermediates
    materialized once on the first compute hart. native: unsharded Linears
    kept on the reduce hart. classify(device, segment): the
    name of the graph class of a segment. quantized_partials: row Linears
    whose per-rank outputs the recipe quantizes before reduction. units: the
    channels a column Linear deals out at a time (the heads that share a
    key head), 32 (an MXINT8 block) for the others; a column Linear of one
    unit is computed whole by every rank. classes: the hart class
    of each hart, the reduce hart first. sink: an operation of a hart that
    reads only the hart's own inputs runs in the first segment of its hart
    that reads it, and the hart of a Linear quantizes its activation."""
    replicated = set(replicated)
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    body = [
        node for node in graph.body if not isinstance(node, (PlaceholderOp, OutputOp))
    ]
    (output,) = [node for node in graph.body if isinstance(node, OutputOp)]

    # ===------------------------------------------------------------------===
    # The sharded values with the pieces each rank holds along their sharded
    # axis, and the Linears split along K (rows): those that read a sharded
    # value. A column Linear deals its MXINT8 blocks out to the ranks, the
    # first ones one more when they do not divide evenly.
    # ===------------------------------------------------------------------===
    # A state deals out its sharded axis the same way, the first ranks one
    # more.
    axes, pieces = {}, {}
    for name, axis in states.items():
        n = graph.node_table[name].tensor_meta["shape"][axis]
        base, extra = divmod(n, count)
        axes[name] = axis
        pieces[name] = [
            [(r * base + min(r, extra), (r + 1) * base + min(r + 1, extra))]
            for r in range(count)
        ]
    rows, tops = set(), set()
    for node in body:
        if node.name in replicated:
            continue
        if isinstance(node, TopkOp) and node.args[0] in axes:
            shape = graph.node_table[node.args[0]].tensor_meta["shape"]
            if node.args[2] % len(shape) != axes[node.args[0]]:
                raise ValueError(f"{node.name}: a top-k off the sharded axis")
            tops.add(node.name)
        elif restoring(graph, node) and node.args[0] in axes:
            if (
                node.name in columns
                or axes[node.args[0]] != len(node.tensor_meta["shape"]) - 1
            ):
                raise ValueError(f"{node.name}: reads a value sharded off its K axis")
            rows.add(node.name)
        elif node.name in columns:
            n = node.tensor_meta["shape"][-1]
            count_segments = segments.get(node.name, 1)
            segment = n // count_segments
            unit = units.get(node.name, 32)
            if segment % unit or unit % 32:
                raise ValueError(
                    f"{node.name}: {segment} channels are not units of {unit} in MXINT8 blocks"
                )
            base, extra = divmod(segment // unit, count)
            held = []
            for r in range(count):
                low = unit * (r * base + min(r, extra))
                high = low + unit * (base + (r < extra))
                if segment == unit:
                    # One unit (the keys or values of the one key head): every
                    # rank computes it whole.
                    low, high = 0, unit
                held.append(
                    [
                        (i * segment + low, i * segment + high)
                        for i in range(count_segments)
                    ]
                )
            axes[node.name] = len(node.tensor_meta["shape"]) - 1
            pieces[node.name] = held
        elif any(name in axes for name in node.parents):
            axes[node.name], pieces[node.name] = follow(graph, node, axes, pieces)
    for name in axes:
        for child in graph.node_table[name].children:
            reader = graph.node_table[child]
            if (
                name in states
                and isinstance(reader, SymSizeOp)
                and reader.args[1] % len(graph.node_table[name].tensor_meta["shape"])
                != axes[name]
            ):
                continue
            if child not in axes.keys() | rows | tops and child != output.name:
                raise ValueError(f"{child} uses the sharded {name} whole")
    # A projection that is not sharded still runs on a compute hart.
    whole = {
        node.name
        for node in body
        if restoring(graph, node)
        and node.name not in replicated | columns | rows | native
    }
    if not set(quantized_partials) <= rows:
        raise ValueError("Partial-output quantization requires row-parallel Linears")
    # The operands the split Linears read (each rank its own) and the
    # restored weights behind them.
    weights = {
        restoring(graph, graph.node_table[name])[1].name
        for name in columns | rows | whole
    }
    restorations = {
        restoring(graph, graph.node_table[name])[0].name
        for name in columns | rows | whole
    }
    for name in restorations:
        restored = graph.node_table[name]
        if (
            not isinstance(
                restored,
                (MXINT8ToFP32DequantOp, MXINT5ToFP32DequantOp, MXFP4ToFP32DequantOp),
            )
            or restored.args[1] != 1
        ):
            raise ValueError(f"{name}: a split weight is not block-quantized along K")
    restored_inputs = {
        node.name
        for node in body
        if isinstance(
            node, (MXINT8ToFP32DequantOp, MXINT5ToFP32DequantOp, MXFP4ToFP32DequantOp)
        )
        and all(child in columns | whole for child in node.children)
    }
    # With `sink`, the hart of a Linear quantizes its activation itself from
    # the FP32 values (its codes are not passed between segments).
    requantized = {
        node.name
        for node in body
        if sink
        and isinstance(node, FP32ToMXINT8QuantOp)
        and node.children
        and all(child in restored_inputs for child in node.children)
    }
    # The reshapes of the activation before its quantization go with it:
    # the hart receives the activation itself, once.
    for name in list(requantized):
        view = graph.node_table[graph.node_table[name].args[0]]
        while isinstance(view, ViewOp) and all(c in requantized for c in view.children):
            requantized.add(view.name)
            view = graph.node_table[view.args[0]]

    def restore(source, name, operations):
        """The decoding `source` of a Linear's activation as `name`, after
        its quantization (and the reshapes before it) when the hart
        quantizes it itself."""
        arguments = list(source.args)
        if source.args[0] in requantized:
            chain = [graph.node_table[source.args[0]]]
            while chain[0].args[0] in requantized:
                chain.insert(0, graph.node_table[chain[0].args[0]])
            previous = chain[0].args[0]
            for node in chain:
                copy = copied(node, f"{node.name}_{name}", [previous, *node.args[1:]])
                operations.append((key, copy))
                previous = copy.name
            arguments[0] = previous
        operations.append((key, copied(source, name, arguments)))

    def local(node, rank):
        """The shape of rank `rank`'s piece of the sharded `node`."""
        shape = list(node.tensor_meta["shape"])
        width = total(pieces[node.name][rank])
        if (
            isinstance(node, FP32ToMXINT8QuantOp)
            and node.args[1] % len(shape) == axes[node.name]
        ) or (
            isinstance(node, MXINT8ActivationOp)
            and node.args[3] % len(shape) == axes[node.name]
        ):
            width += width // 32
        shape[axes[node.name]] = width
        return shape

    # ===------------------------------------------------------------------===
    # The operations of each hart, each keyed by the graph position of the
    # operation it comes from, with the sliced weights and states as their
    # parameters and inputs.
    # ===------------------------------------------------------------------===
    operations = [[] for _ in range(count + 1)]
    placeholders = [[] for _ in range(count + 1)]
    local_values = [{} for _ in range(count + 1)]
    position = {node.name: index for index, node in enumerate(graph.body)}
    decoded = set()

    def renamed(argument, rank, slices={}):
        if isinstance(argument, list):
            return [renamed(element, rank, slices) for element in argument]
        if isinstance(argument, str) and (argument in axes or argument in weights):
            return f"{argument}_r{rank}"
        return slices.get(argument, argument)

    for name in states:
        state = graph.node_table[name]
        for rank in range(count):
            placeholders[rank + 1].append(
                (
                    copied(state, f"{name}_r{rank}", [], local(state, rank)),
                    NodeType.InputNode,
                )
            )

    for node in body:
        key = position[node.name]
        if (
            node.name in weights
            or node.name in restorations
            or node.name in restored_inputs
            or node.name in requantized
            or node.name in replicated
        ):
            continue
        if node.name in whole:
            arguments = list(node.args)
            restored, operand = restoring(graph, node)
            for index in (0, 1):
                source = graph.node_table[node.args[index]]
                if index == 1 and operand is not restored:
                    # The transpose of the restored weight a product reads.
                    name = f"{restored.name}_{node.name}_whole"
                    operations[1].append(
                        (key, copied(restored, name, list(restored.args)))
                    )
                    arguments[1] = f"{operand.name}_{node.name}_whole"
                    operations[1].append(
                        (key, copied(operand, arguments[1], [name, *operand.args[1:]]))
                    )
                elif isinstance(
                    source,
                    (
                        MXINT8ToFP32DequantOp,
                        MXINT5ToFP32DequantOp,
                        MXFP4ToFP32DequantOp,
                    ),
                ):
                    name = f"{source.name}_{node.name}_whole"
                    restore(source, name, operations[1])
                    arguments[index] = name
            operations[1].append((key, copied(node, node.name, arguments)))
        elif node.name in columns or node.name in rows:
            restored, operand = restoring(graph, node)
            weight = graph.node_table[restored.args[0]]
            packed = values[weight.name]
            n, k = restored.tensor_meta["shape"]
            for rank in range(count):
                if node.name in columns:
                    part = torch.cat([packed[s:e] for s, e in pieces[node.name][rank]])
                    shape = [part.shape[0], k]
                    source = graph.node_table[node.args[0]]
                    arguments = [
                        source.name,
                        f"{operand.name}_r{rank}",
                        *node.args[2:],
                    ]
                    if source.name in restored_inputs:
                        arguments[0] = f"{source.name}_r{rank}"
                        # Linears that read one activation restore it once.
                        if arguments[0] not in decoded:
                            decoded.add(arguments[0])
                            restore(source, arguments[0], operations[rank + 1])
                    result = local(node, rank)
                else:
                    ((low, high),) = pieces[node.args[0]][rank]
                    if low % 32 or high % 32:
                        raise ValueError(
                            f"{node.name}: rank {rank} K {low}..{high} splits MXINT8 blocks"
                        )
                    if isinstance(restored, MXFP4ToFP32DequantOp):
                        part = torch.cat(
                            [
                                packed[:, low // 2 : high // 2],
                                packed[:, k // 2 + low // 32 : k // 2 + high // 32],
                            ],
                            1,
                        )
                    elif isinstance(restored, MXINT5ToFP32DequantOp):
                        codes = k * 5 // 8
                        part = torch.cat(
                            [
                                packed[:, low * 5 // 8 : high * 5 // 8],
                                packed[:, codes + low // 32 : codes + high // 32],
                                packed[
                                    :,
                                    codes
                                    + k // 32
                                    + low // 32 : codes
                                    + k // 32
                                    + high // 32,
                                ],
                            ],
                            1,
                        )
                    else:
                        part = torch.cat(
                            [
                                packed[:, low:high],
                                packed[:, k + low // 32 : k + high // 32],
                            ],
                            1,
                        )
                    shape = [n, high - low]
                    arguments = renamed(list(node.args), rank)
                    result = node.tensor_meta["shape"]
                placeholder = copied(weight, f"{weight.name}_r{rank}", [], part.shape)
                local_values[rank + 1][placeholder.name] = part
                placeholders[rank + 1].append((placeholder, NodeType.FakeNode))
                operations[rank + 1].append(
                    (
                        key,
                        copied(
                            restored,
                            f"{restored.name}_r{rank}",
                            [placeholder.name, 1],
                            shape,
                        ),
                    )
                )
                if operand is not restored:
                    # A product reads the transpose of the restored weight.
                    operations[rank + 1].append(
                        (
                            key,
                            copied(
                                operand,
                                f"{operand.name}_r{rank}",
                                [f"{restored.name}_r{rank}", *operand.args[1:]],
                                shape[::-1],
                            ),
                        )
                    )
                # With one rank the product of a row Linear is its whole sum.
                name = (
                    node.name
                    if node.name in rows and count == 1
                    else f"{node.name}_r{rank}"
                )
                if node.name in quantized_partials:
                    shape = list(result)
                    if shape[-1] % 32:
                        raise ValueError(
                            f"{name}: quantized partial requires complete output blocks"
                        )
                    raw = name + "_partial"
                    operations[rank + 1].append(
                        (key, copied(node, raw, arguments, result))
                    )
                    quantized = FP32ToMXINT8QuantOp()
                    quantized.name = name + "_mxint8"
                    quantized.add_argument(raw)
                    quantized.add_parent(raw)
                    quantized.add_argument(len(shape) - 1)
                    shape[-1] += shape[-1] // 32
                    quantized.tensor_meta = {"shape": shape, "dtype": TensorDType.Int8}
                    operations[rank + 1].append((key, quantized))
                    decoded_partial = MXINT8ToFP32DequantOp()
                    decoded_partial.name = name
                    decoded_partial.add_argument(quantized.name)
                    decoded_partial.add_parent(quantized.name)
                    decoded_partial.add_argument(len(shape) - 1)
                    decoded_partial.tensor_meta = dict(node.tensor_meta)
                    operations[rank + 1].append((key, decoded_partial))
                else:
                    operations[rank + 1].append(
                        (key, copied(node, name, arguments, result))
                    )
            if node.name in rows:
                running = f"{node.name}_r0"
                for rank in range(1, count):
                    add = AddOp()
                    add.name = (
                        node.name if rank == count - 1 else f"{node.name}_sum{rank}"
                    )
                    for name in (running, f"{node.name}_r{rank}"):
                        add.add_argument(name)
                        add.add_parent(name)
                    add.tensor_meta = {
                        "shape": list(node.tensor_meta["shape"]),
                        "dtype": node.tensor_meta["dtype"],
                    }
                    operations[0].append((key, add))
                    running = add.name
        elif node.name in tops:
            # Each rank: the k largest values of its piece and their indices
            # there. The reduce hart: the values and the indices of the ranks
            # joined in rank order, each index offset by the start of the
            # rank's piece.
            source = node.args[0]
            items = {graph.node_table[name].args[1]: name for name in node.children}
            parts = {index: [] for index in items}
            for rank in range(count):
                top = copied(
                    node, f"{node.name}_r{rank}", renamed(list(node.args), rank)
                )
                operations[rank + 1].append((key, top))
                for index, name in items.items():
                    item = copied(
                        graph.node_table[name], f"{name}_r{rank}", [top.name, index]
                    )
                    operations[rank + 1].append((key, item))
                    parts[index].append(item.name)
                ((low, _),) = pieces[source][rank]
                if low:
                    offset = AddOp()
                    offset.name = f"{items[1]}_offset_r{rank}"
                    for argument in (f"{items[1]}_r{rank}", low):
                        offset.add_argument(argument)
                    offset.add_parent(f"{items[1]}_r{rank}")
                    offset.tensor_meta = dict(graph.node_table[items[1]].tensor_meta)
                    operations[0].append((key, offset))
                    parts[1][-1] = offset.name
            for index, name in items.items():
                meta = graph.node_table[name].tensor_meta
                shape = list(meta["shape"])
                shape[axes[source]] *= count
                joined = CatOp()
                joined.name = name
                joined.add_argument(parts[index])
                joined.add_argument(axes[source])
                for part in parts[index]:
                    joined.add_parent(part)
                joined.tensor_meta = {"shape": shape, "dtype": meta["dtype"]}
                operations[0].append((key, joined))
        elif isinstance(node, GetItemOp) and node.args[0] in tops:
            continue
        elif node.name in axes:
            spans = spanned(graph, node, axes) if "shape" in node.tensor_meta else {}
            for rank in range(count):
                # The unsharded arguments across the sharded axis: the rank
                # reads its pieces of them.
                slices = {}
                for name, axis in spans.items():
                    source = graph.node_table[name]
                    # A parameter is sliced offline: each rank its own.
                    if name in values:
                        part = torch.cat(
                            [
                                values[name].narrow(axis, low, high - low)
                                for low, high in pieces[node.name][rank]
                            ],
                            axis,
                        )
                        placeholder = copied(
                            source, f"{name}_{node.name}_r{rank}", [], part.shape
                        )
                        local_values[rank + 1][placeholder.name] = part
                        placeholders[rank + 1].append((placeholder, NodeType.FakeNode))
                        slices[name] = placeholder.name
                        continue
                    # A value of the reduce hart is sliced there: each rank
                    # receives its pieces only. A value the same at every
                    # offset along the axis (a mask broadcast over the
                    # heads): each rank reads its pieces from the start, as
                    # every other rank.
                    hart = rank + 1
                    if name not in replicated and not isinstance(source, PlaceholderOp):
                        hart = 0
                    same = uniform(graph, name, axis)
                    parts = []
                    for index, (low, high) in enumerate(pieces[node.name][rank]):
                        narrowed = narrow_broadcast(
                            graph,
                            source,
                            axis,
                            high - low,
                            f"_{node.name}_r{rank}_{index}",
                        )
                        if narrowed:
                            operations[hart].extend((key, n) for n in narrowed)
                            parts.append(narrowed[-1])
                            continue
                        if same:
                            low, high = 0, high - low
                        if (low, high) == (0, source.tensor_meta["shape"][axis]):
                            # The whole value (beside keys every rank computes
                            # whole): the rank reads it where it is.
                            parts.append(source)
                            continue
                        piece = SliceOp()
                        piece.name = f"{name}_{node.name}_r{rank}_{index}"
                        for argument in (name, axis, low, high):
                            piece.add_argument(argument)
                        piece.add_parent(name)
                        shape = list(source.tensor_meta["shape"])
                        shape[axis] = high - low
                        piece.tensor_meta = {
                            "shape": shape,
                            "dtype": source.tensor_meta["dtype"],
                        }
                        operations[hart].append((key, piece))
                        parts.append(piece)
                    if len(parts) > 1:
                        joined = CatOp()
                        joined.name = f"{name}_{node.name}_r{rank}"
                        joined.add_argument([part.name for part in parts])
                        joined.add_argument(axis)
                        for part in parts:
                            joined.add_parent(part.name)
                        shape = list(source.tensor_meta["shape"])
                        shape[axis] = total(pieces[node.name][rank])
                        joined.tensor_meta = {
                            "shape": shape,
                            "dtype": source.tensor_meta["dtype"],
                        }
                        operations[hart].append((key, joined))
                        parts = [joined]
                    slices[name] = parts[0].name
                arguments = renamed(list(node.args), rank, slices)
                if isinstance(node, (UnbindOp, SplitOp)):
                    # Each result: the rank's pieces along the sharded axis,
                    # within its chunk when the split cuts that axis.
                    axis = axes[node.name]
                    along = (
                        isinstance(node, SplitOp)
                        and node.args[2]
                        % len(graph.node_table[node.args[0]].tensor_meta["shape"])
                        == axis
                    )
                    if along:
                        held = [
                            p for p in pieces[node.name][rank] if p[0] < node.args[1]
                        ]
                        arguments[1] = total(held)
                    shapes = []
                    for index, shape in enumerate(node.tensor_meta["shape"]):
                        shape = list(shape)
                        if isinstance(node, UnbindOp):
                            result_axis = axis - (
                                node.args[1] % (len(shape) + 1) < axis
                            )
                        else:
                            result_axis = axis
                        width = total(pieces[node.name][rank])
                        if along:
                            low, high = index * node.args[1], (index + 1) * node.args[1]
                            width = sum(
                                min(e, high) - max(s, low)
                                for s, e in pieces[node.name][rank]
                                if s < high and e > low
                            )
                        shape[result_axis] = width
                        shapes.append(shape)
                    operations[rank + 1].append(
                        (key, copied(node, f"{node.name}_r{rank}", arguments, shapes))
                    )
                    continue
                shape = local(node, rank)
                if isinstance(node, (ViewOp, ExpandOp)):
                    arguments[1] = list(arguments[1])
                    if arguments[1][axes[node.name]] != -1:
                        arguments[1][axes[node.name]] = shape[axes[node.name]]
                operations[rank + 1].append(
                    (key, copied(node, f"{node.name}_r{rank}", arguments, shape))
                )
        else:
            operations[0].append((key, copied(node, node.name, list(node.args))))

    # ===------------------------------------------------------------------===
    # The outputs: hart 0 joins the slices the ranks hold of every sharded
    # one. A replicated output is materialized once on the first compute
    # hart; its consumers share that value instead of recomputing it.
    # ===------------------------------------------------------------------===
    joins = set()
    for name in dict.fromkeys([*output.args, *sorted(shared)]):
        if name in shared and name not in replicated:
            raise ValueError(f"{name}: a shared value must be replicated")
        if name in replicated:
            node = graph.node_table[name]
            replicated.remove(name)
            operations[1].append((position[name], copied(node, name, list(node.args))))
            joins.add(name)
            continue
        if name not in axes:
            continue
        held = pieces[name]
        meta = graph.node_table[name].tensor_meta
        # A value every rank holds whole (a column Linear of one unit and
        # what follows it) is the first rank's.
        whole = all(rank == [(0, meta["shape"][axes[name]])] for rank in held)
        if not whole and (
            any(len(rank) != 1 for rank in held)
            or [rank[0][0] for rank in held] != [0] + [rank[0][1] for rank in held[:-1]]
        ):
            raise ValueError(
                f"{name}: the slices of the ranks do not join in rank order"
            )
        joined = CatOp()
        joined.name = name
        parts = [f"{name}_r{rank}" for rank in range(1 if whole else count)]
        joined.add_argument(parts)
        joined.add_argument(axes[name])
        for part in parts:
            joined.add_parent(part)
        joined.tensor_meta = {"shape": list(meta["shape"]), "dtype": meta["dtype"]}
        operations[0].append((len(graph.body), joined))
        joins.add(name)

    if per_rank:
        # Materialize a replicated preparation region once on each consuming hart.
        prefix = set(per_rank)
        frontier = list(prefix)
        while frontier:
            name = frontier.pop()
            if name not in replicated:
                raise ValueError(f"{name}: per-rank preparation must be replicated")
            node = graph.node_table[name]
            for child in node.children:
                if child in replicated and child not in prefix:
                    prefix.add(child)
                    frontier.append(child)
        for hart, ops in enumerate(operations):
            needed = set()
            frontier = [parent for _, node in ops for parent in node.parents]
            while frontier:
                name = frontier.pop()
                if name in prefix and name not in needed:
                    needed.add(name)
                    frontier.extend(graph.node_table[name].parents)
            aliases = {name: f"{name}_local_r{hart}" for name in needed}

            def substitute(argument):
                if isinstance(argument, list):
                    return [substitute(value) for value in argument]
                return (
                    aliases.get(argument, argument)
                    if isinstance(argument, str)
                    else argument
                )

            operations[hart] = [
                (key, copied(node, node.name, substitute(node.args)))
                for key, node in ops
            ] + [
                (
                    position[name],
                    copied(
                        graph.node_table[name],
                        aliases[name],
                        substitute(graph.node_table[name].args),
                    ),
                )
                for name in sorted(needed, key=position.__getitem__)
            ]
            operations[hart].sort(key=lambda item: item[0])
        replicated -= prefix

    parameters = {node.name: (node, values[node.name]) for node in graph.params}
    inputs = {node.name: node for node in graph.inputs}
    slices = {}
    for hart in range(1, count + 1):
        for node, kind in placeholders[hart]:
            if kind == NodeType.FakeNode:
                parameters[node.name] = (node, local_values[hart][node.name])
            else:
                state = node.name.rsplit("_r", 1)[0]
                ((low, high),) = pieces[state][hart - 1]
                inputs[node.name] = node
                slices[node.name] = (state, axes[state], low, high)
    return Device(
        graph,
        count,
        operations,
        parameters,
        inputs,
        slices,
        replicated,
        output,
        position,
        joins,
        classify,
        classes,
        sink,
    )


class Segment:
    """Operations of one hart between two receptions, or the outputs the
    reduce hart computes there (`join`): an instance of `graph`, the graph of its class
    `graph_class`, which the device program launches on a hart of
    `hart_class`."""

    def __init__(self, hart, hart_class, join):
        self.hart = hart
        self.hart_class = hart_class
        self.join = join
        self.name = None
        self.graph_class = None
        self.graph = None
        self.ops = []
        self.params = []
        self.inputs = []
        self.outputs = []


def tensor_type(meta):
    """The MLIR tensor type of a value: a symbolic size (the token count) is
    dynamic."""
    sizes = ["?" if isinstance(d, torch.SymInt) else str(int(d)) for d in meta["shape"]]
    with ir.Context():
        element = str(mlir_element_type_get(meta["dtype"]))
    return f"tensor<{'x'.join(sizes + [element])}>"


class Device:
    """The partitioned graph as a device program: the operations of each hart
    cut into segments where they receive a value of another hart, each
    segment a graph launched on the class of its hart (`classes`, the reduce
    hart first) and the program that launches them in an order
    their values allow, on the reduce hart. Placement on the harts and the
    transfers between them are the device compiler's.

    Every segment computes the replicated operations it uses itself. The
    program takes the parameters packed into one buffer per element type,
    then the inputs of the graph; it hands each rank its slice of every state
    and returns the outputs of the graph, which hart 0 computes or joins."""

    def __init__(
        self,
        graph,
        ranks,
        operations,
        parameters,
        inputs,
        slices,
        replicated,
        output,
        position,
        joins,
        classify,
        classes,
        sink,
    ):
        self.graph = graph
        self.ranks = ranks
        self.classes = classes
        self.parameters = parameters
        self.inputs = inputs
        self.slices = slices
        self.output = output
        producer = {
            node.name: hart for hart, ops in enumerate(operations) for _, node in ops
        }
        self.meta = {
            node.name: node.tensor_meta for ops in operations for _, node in ops
        }
        self.meta.update({name: node.tensor_meta for name, node in inputs.items()})

        # The computed values each operation reads: through the replicated
        # operations, which every segment computes itself, those they read.
        nodes = {node.name: node for ops in operations for _, node in ops}
        through = {}

        def reads(names):
            result = []
            for name in names:
                if name in replicated:
                    if name not in through:
                        through[name] = reads(graph.node_table[name].parents)
                    result += through[name]
                elif name in producer:
                    result.append(name)
            return result

        # The depth of every operation: the receptions on the longest path to
        # it. The operations of a hart at one depth form a segment: a value
        # of another hart comes from a smaller depth, so the segments depend
        # on each other without a cycle.
        depth = {}
        for name in nodes:
            stack = [name]
            while stack:
                current = stack[-1]
                if current in depth:
                    stack.pop()
                    continue
                parents = reads(nodes[current].parents)
                pending = [p for p in parents if p not in depth]
                if pending:
                    stack.extend(pending)
                    continue
                depth[current] = max(
                    [depth[p] + (producer[p] != producer[current]) for p in parents],
                    default=0,
                )
                stack.pop()
        # An operation that reads no computed value (the restoring of a
        # weight) joins the segment of its first reader on its hart, no later
        # than the segment before the first reader on another hart; with
        # `sink`, so does one that reads only its hart's own inputs (the
        # quantization of a rank's slice of the KV cache), all its readers
        # on its hart: it runs where it is read, its value never passed
        # between two segments.
        readers = {}
        for name, node in nodes.items():
            for parent in node.parents:
                readers.setdefault(parent, []).append(name)
        for hart, ops in enumerate(operations):
            for _, node in reversed(ops):
                own = (
                    sink
                    and depth[node.name] == 0
                    and readers.get(node.name)
                    and all(producer[r] == hart for r in readers[node.name])
                )
                if not reads(node.parents) or own:
                    depth[node.name] = max(
                        0,
                        min(
                            (
                                depth[r] - (producer[r] != hart)
                                for r in readers.get(node.name, [])
                            ),
                            default=0,
                        ),
                    )
        segments = []
        for hart, ops in enumerate(operations):
            levels = {}
            for key, node in ops:
                level = (depth[node.name], node.name in joins)
                if level not in levels:
                    levels[level] = Segment(hart, self.classes[hart], level[1])
                    segments.append(levels[level])
                levels[level].ops.append((key, node))
        owner = {node.name: segment for segment in segments for _, node in segment.ops}

        # Keep decoding and its layout views with their consumers. Passing a
        # decoded FP32 tensor across segments hides the quantized matrix operand.
        operation_keys = {node.name: key for ops in operations for key, node in ops}
        for segment in segments:
            copied_layouts = []
            defined = {node.name for _, node in segment.ops}
            pending = [name for _, node in segment.ops for name in node.parents]
            while pending:
                name = pending.pop()
                if name in defined or name not in nodes:
                    continue
                chain = []
                node = nodes[name]
                while isinstance(
                    node, (ViewOp, PermuteOp, ExpandOp, UnsqueezeOp, CloneOp, SliceOp)
                ):
                    chain.append(node)
                    parent = node.args[0]
                    if parent in nodes:
                        node = nodes[parent]
                    elif parent in replicated:
                        node = graph.node_table[parent]
                    else:
                        break
                if not isinstance(
                    node,
                    (
                        MXINT8ToFP32DequantOp,
                        MXINT5ToFP32DequantOp,
                        MXFP4ToFP32DequantOp,
                    ),
                ):
                    # A reshape of a value that leaves its segment anyway (as
                    # it is, or as a copy of it: an output, or read by another
                    # segment) is made where it is read, from what leaves:
                    # the value is not passed a second time in another shape
                    # (two results of one buffer).
                    reshapes = []
                    base = nodes[name]
                    while (
                        isinstance(base, (ViewOp, UnsqueezeOp))
                        or isinstance(base, ExpandOp)
                        and base.args[0] in nodes
                        and list(base.tensor_meta["shape"])
                        == list(nodes[base.args[0]].tensor_meta["shape"])
                    ) and base.args[0] in nodes:
                        reshapes.append(base)
                        base = nodes[base.args[0]]
                    names = {view.name for view in reshapes}

                    def leaves(value):
                        return value.name in output.args or any(
                            reader not in names
                            and owner[reader] is not owner[value.name]
                            for reader in readers.get(value.name, [])
                        )

                    leaving = (
                        [base]
                        if leaves(base)
                        else [
                            nodes[reader]
                            for reader in readers.get(base.name, [])
                            if isinstance(nodes[reader], CloneOp)
                            and owner[reader] is owner[base.name]
                            and leaves(nodes[reader])
                        ]
                    )
                    if reshapes and leaving and owner[base.name] is not segment:
                        for original in reversed(reshapes):
                            if original.name in defined:
                                continue
                            arguments = list(original.args)
                            if arguments[0] == base.name:
                                arguments[0] = leaving[0].name
                            copied_layouts.append(
                                (
                                    operation_keys[original.name],
                                    copied(original, original.name, arguments),
                                )
                            )
                            defined.add(original.name)
                    continue
                for original in reversed([*chain, node]):
                    if original.name in defined or original.name in replicated:
                        continue
                    copied_layouts.append(
                        (
                            operation_keys[original.name],
                            copied(original, original.name, list(original.args)),
                        )
                    )
                    defined.add(original.name)
                    pending.extend(original.parents)
            segment.ops = copied_layouts + segment.ops

        # The replicated operations each segment uses, its parameters and
        # its inputs: values of other segments and of the program.
        for segment in segments:
            needed = set()
            frontier = [name for _, node in segment.ops for name in node.parents]
            while frontier:
                name = frontier.pop()
                if name in replicated and name not in needed:
                    needed.add(name)
                    frontier.extend(graph.node_table[name].parents)
            copies = [
                (
                    position[name],
                    copied(
                        graph.node_table[name], name, list(graph.node_table[name].args)
                    ),
                )
                for name in needed
            ]
            segment.ops = sorted(copies + segment.ops, key=lambda entry: entry[0])
            defined = {node.name for _, node in segment.ops}
            for _, node in segment.ops:
                for name in node.parents:
                    if (
                        name in defined
                        or name in segment.params
                        or name in segment.inputs
                    ):
                        continue
                    if name in parameters:
                        segment.params.append(name)
                    elif name in owner or name in inputs:
                        if "shape" not in self.meta[name]:
                            raise ValueError(
                                f"{name}: a value without a shape crosses segments"
                            )
                        segment.inputs.append(name)
                    else:
                        raise ValueError(
                            f"{node.name}: reads {name}, which no segment computes"
                        )
        for segment in segments:
            for name in segment.inputs:
                if name in owner and name not in owner[name].outputs:
                    owner[name].outputs.append(name)
        for name in output.args:
            if name not in owner[name].outputs:
                owner[name].outputs.append(name)
        for segment in segments:
            order = {node.name: index for index, (_, node) in enumerate(segment.ops)}
            segment.outputs.sort(key=order.__getitem__)

        # The program order: each segment after those it reads, the earliest
        # in the graph first.
        waiting = {
            id(segment): {id(owner[name]) for name in segment.inputs if name in owner}
            for segment in segments
        }
        first = {
            id(s): min(key for key, node in s.ops if node.name in owner)
            for s in segments
        }
        self.segments = []
        while waiting:
            ready = [s for s in segments if id(s) in waiting and not waiting[id(s)]]
            if not ready:
                names = {
                    id(s): f"hart {s.hart} from {s.ops[0][1].name}" for s in segments
                }
                raise ValueError(
                    "the segments depend on each other in a cycle: "
                    + "; ".join(
                        f"{names[key]} waits for {sorted(names[d] for d in waits)}"
                        for key, waits in waiting.items()
                    )
                )
            chosen = min(ready, key=lambda s: (first[id(s)], s.hart))
            del waiting[id(chosen)]
            for dependencies in waiting.values():
                dependencies.discard(id(chosen))
            self.segments.append(chosen)
        counts = {}
        for segment in self.segments:
            index = counts.get(segment.hart, 0)
            counts[segment.hart] = index + 1
            segment.name = (
                f"reduce{index}"
                if segment.hart == 0
                else f"rank{segment.hart - 1}_{index}"
            )

        # The graph of every segment: one per class, of which the segments
        # are instances that compute the same up to the names of their
        # values and their parameters. A class whose instances differ (an
        # uneven split, a layer of another kind) has a graph per variant,
        # <class>_<n>.
        self.graphs = {}
        variants = {}
        # The segment that computes each value, and the segments that read
        # each segment's outputs: what the recipe names classes by.
        self.owner = owner
        self.nodes = nodes
        for segment in self.segments:
            segment.readers = []
        for segment in self.segments:
            for name in segment.inputs:
                if name in owner and segment not in owner[name].readers:
                    owner[name].readers.append(segment)
        for segment in self.segments:
            segment.graph_class = classify(self, segment)
            form = self.form(segment)
            known = variants.setdefault(segment.graph_class, [])
            for other, name in known:
                if other == form:
                    segment.graph = name
                    break
            else:
                segment.graph = (
                    f"{segment.graph_class}_{len(known)}"
                    if known
                    else segment.graph_class
                )
                known.append((form, segment.graph))
                self.graphs[segment.graph] = segment

        # The parameters, packed by element type in the order the segments
        # read them.
        self.packs = {}
        self.offsets = {}
        for segment in self.segments:
            for name in segment.params:
                if name in self.offsets:
                    continue
                node, _ = parameters[name]
                element = tensor_type(
                    {"shape": [], "dtype": node.tensor_meta["dtype"]}
                )[7:-1]
                count = math.prod(int(d) for d in node.tensor_meta["shape"])
                offset = self.packs.get(element, 0)
                self.offsets[name] = (element, offset, count)
                self.packs[element] = offset + count

    def packed(self):
        """The packed parameters, one array per element type."""
        arrays = {element: [] for element in self.packs}
        for name, (element, _, _) in self.offsets.items():
            arrays[element].append(
                self.parameters[name][1].detach().cpu().numpy().reshape(-1)
            )
        return {element: numpy.concatenate(parts) for element, parts in arrays.items()}

    def form(self, segment):
        """What a segment computes, up to the names of its values: its hart
        class, the types of its parameters and inputs, its operations over
        them and its outputs."""
        local = {name: f"%param{i}" for i, name in enumerate(segment.params)}
        local.update({name: f"%input{i}" for i, name in enumerate(segment.inputs)})

        def canonical(value):
            if isinstance(value, str):
                return local.get(value, value)
            if isinstance(value, (list, tuple)):
                return [canonical(v) for v in value]
            if isinstance(value, dict):
                return {key: canonical(v) for key, v in value.items()}
            return repr(value)

        ops = []
        for _, node in segment.ops:
            ops.append(
                (
                    type(node).__name__,
                    canonical(list(node.args)),
                    canonical(dict(node.kwargs)),
                    repr(node.tensor_meta),
                )
            )
            local[node.name] = f"%op{len(ops) - 1}"
        return (
            segment.hart_class,
            [
                tensor_type(self.parameters[name][0].tensor_meta)
                for name in segment.params
            ],
            [tensor_type(self.meta[name]) for name in segment.inputs],
            ops,
            [local[name] for name in segment.outputs],
        )

    def flat(self, name):
        """The type of a parameter as the program passes it: its slice of
        the pack of its element type."""
        element, _, count = self.offsets[name]
        return f"tensor<{count}x{element}>"

    def layout(self, name):
        """The layout of the input `name` of a graph: a rank's slice of a
        state that is not contiguous in it (its axis inside another) is read
        in place, with the strides of the state."""
        if name not in self.slices:
            return None
        state, axis, _, _ = self.slices[name]
        shape = self.meta[state]["shape"]
        if all(d == 1 for d in shape[:axis]):
            return None
        shape = [int(d) for d in shape]
        strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
        return f"strided<[{', '.join(map(str, strides))}], offset: ?>"

    def signature(self, segment):
        """The argument and result types of the graph of a segment: its
        parameters, each as a slice of a pack, its inputs; its outputs."""
        arguments = [self.flat(name) for name in segment.params]
        arguments += [tensor_type(self.meta[name]) for name in segment.inputs]
        return arguments, [tensor_type(self.meta[name]) for name in segment.outputs]

    def declarations(self):
        """The declarations of the graphs of the segments, as MLIR lines."""
        lines = []
        for name, segment in self.graphs.items():
            arguments, results = self.signature(segment)
            count = len(segment.params)
            for i, input_ in enumerate(segment.inputs):
                if self.layout(input_):
                    arguments[
                        count + i
                    ] += f" {{bufferization.buffer_layout = {self.layout(input_)}}}"
            lines.append(
                f"func.func private @{name}({', '.join(arguments)}) -> ({', '.join(results)})"
            )
        return lines

    def launches(self, tag, packs, inputs):
        """The launches of the segments as lines of a device program, every
        value named with the prefix `tag`: the parameters sliced from the
        packs `packs` (element type -> value), the inputs of the graph
        `inputs` (name -> value). Returns the lines and the values of the
        outputs of the graph."""
        body = []
        values = dict(inputs)
        # Each symbolic size, from an input that has it as a size; a size
        # that is an expression of them (the KV length after the tokens),
        # computed with arith.
        sizes = {}
        for name in inputs:
            for axis, dimension in enumerate(self.meta[name]["shape"]):
                if (
                    isinstance(dimension, torch.SymInt)
                    and dimension.node.expr.is_Symbol
                    and dimension.node.expr not in sizes
                ):
                    body.append(
                        f"%{tag}axis_{name}_{axis} = arith.constant {axis} : index"
                    )
                    body.append(
                        f"%{tag}size_{dimension} = tensor.dim {inputs[name]}, %{tag}axis_{name}_{axis} : {tensor_type(self.meta[name])}"
                    )
                    sizes[dimension.node.expr] = f"%{tag}size_{dimension}"

        size_id = 0

        def size(expression):
            nonlocal size_id
            if expression not in sizes:
                name = f"%{tag}size{size_id}"
                size_id += 1
                if expression.is_Integer:
                    body.append(f"{name} = arith.constant {int(expression)} : index")
                elif expression.func.__name__ == "FloorDiv":
                    left, right = [size(term) for term in expression.args]
                    body.append(f"{name} = arith.divui {left}, {right} : index")
                elif expression.is_Add or expression.is_Mul:
                    terms = [size(term) for term in expression.args]
                    operation = "arith.addi" if expression.is_Add else "arith.muli"
                    for index, term in enumerate(terms[1:]):
                        body.append(
                            f"{name}_{index} = {operation} {terms[0] if index == 0 else f'{name}_{index - 1}'}, {term} : index"
                        )
                    name = f"{name}_{len(terms) - 2}"
                else:
                    raise ValueError(f"no value for the size {expression}")
                sizes[expression] = name
            return sizes[expression]

        for name, (state, axis, low, high) in self.slices.items():
            shape = self.meta[state]["shape"]
            # A rank that holds the whole state reads the state itself.
            if low == 0 and high == shape[axis]:
                values[name] = inputs[state]
                continue
            offsets = [low if i == axis else 0 for i in range(len(shape))]
            extent = [
                (
                    high - low
                    if i == axis
                    else size(d.node.expr) if isinstance(d, torch.SymInt) else d
                )
                for i, d in enumerate(shape)
            ]
            body.append(
                f"%{tag}state_{name} = tensor.extract_slice {inputs[state]}[{', '.join(map(str, offsets))}]"
                f" [{', '.join(map(str, extent))}] [{', '.join('1' for _ in shape)}]"
                f" : {tensor_type(self.meta[state])} to {tensor_type(self.meta[name])}"
            )
            values[name] = f"%{tag}state_{name}"
        # Contiguous output shards can be produced directly in disjoint
        # slices of the final tensor, without a gather task on the control hart.
        joined, destinations = {}, {}
        for segment in self.segments:
            if not segment.join or not all(
                isinstance(node, CatOp)
                and isinstance(node.tensor_meta["shape"][node.args[1]], int)
                and all(
                    isinstance(d, int) and d == 1
                    for d in node.tensor_meta["shape"][: node.args[1]]
                )
                for _, node in segment.ops
            ):
                continue
            for _, node in segment.ops:
                shape = node.tensor_meta["shape"]
                operands = ", ".join(
                    size(d.node.expr) for d in shape if isinstance(d, torch.SymInt)
                )
                joined[node.name] = f"%{tag}joined_{node.name}"
                body.append(
                    f"{joined[node.name]} = tensor.empty({operands})"
                    f" : {tensor_type(node.tensor_meta)}"
                )
                # The only part is the whole tensor.
                if len(node.args[0]) == 1:
                    destinations[node.args[0][0]] = node.name, None
                    continue
                low = 0
                for part in node.args[0]:
                    extent = self.meta[part]["shape"]
                    offsets = [
                        low if i == node.args[1] else 0 for i in range(len(shape))
                    ]
                    dimensions = [
                        size(d.node.expr) if isinstance(d, torch.SymInt) else d
                        for d in extent
                    ]
                    region = (
                        f"[{', '.join(map(str, offsets))}]"
                        f" [{', '.join(map(str, dimensions))}]"
                        f" [{', '.join('1' for _ in shape)}]"
                    )
                    destinations[part] = node.name, region
                    low += extent[node.args[1]]
        for segment in self.segments:
            if segment.join and all(name in joined for name in segment.outputs):
                for name in segment.outputs:
                    values[name] = joined[name]
                continue
            for name in segment.params:
                if name in values:
                    continue
                element, offset, count = self.offsets[name]
                body.append(
                    f"%{tag}param_{name} = tensor.extract_slice {packs[element]}[{offset}] [{count}] [1]"
                    f" : tensor<{self.packs[element]}x{element}> to tensor<{count}x{element}>"
                )
                values[name] = f"%{tag}param_{name}"
            arguments, results = self.signature(segment)
            outs = []
            for index, name in enumerate(segment.outputs):
                operands = ", ".join(
                    size(d.node.expr)
                    for d in self.meta[name]["shape"]
                    if isinstance(d, torch.SymInt)
                )
                if name in destinations and destinations[name][1] is None:
                    outs.append(joined[destinations[name][0]])
                    continue
                if name in destinations:
                    whole, region = destinations[name]
                    body.append(
                        f"%{tag}{segment.name}_out{index} = tensor.extract_slice"
                        f" {joined[whole]}{region} : {tensor_type(self.meta[whole])}"
                        f" to {results[index]}"
                    )
                else:
                    body.append(
                        f"%{tag}{segment.name}_out{index} = tensor.empty({operands}) : {results[index]}"
                    )
                outs.append(f"%{tag}{segment.name}_out{index}")
            many = f":{len(results)}" if len(results) > 1 else ""
            operands = [values[name] for name in segment.params + segment.inputs] + outs
            body.append(
                f'%{tag}{segment.name}{many} = "device.launch"({", ".join(operands)})'
                f' <{{callee = @{segment.graph}, hart_class = "{segment.hart_class}",'
                f" operandSegmentSizes = array<i32: {len(arguments)}, {len(results)}>}}>"
                f" {{device.rank = {segment.hart - 1 if segment.hart else 0} : i64}}"
                f" : ({', '.join(arguments + results)}) -> ({', '.join(results)})"
            )
            for index, name in enumerate(segment.outputs):
                values[name] = (
                    f"%{tag}{segment.name}#{index}" if many else f"%{tag}{segment.name}"
                )
                if name in destinations and destinations[name][1] is None:
                    joined[destinations[name][0]] = values[name]
                elif name in destinations:
                    whole, region = destinations[name]
                    result = f"%{tag}joined_{name}"
                    body.append(
                        f"{result} = tensor.insert_slice {values[name]} into"
                        f" {joined[whole]}{region} : {results[index]}"
                        f" into {tensor_type(self.meta[whole])}"
                    )
                    joined[whole] = result
        return body, [values[name] for name in self.output.args]

    def program(self, function, packs, inputs):
        """The device program `function` as MLIR: a launch of the graph of every
        segment, its parameters sliced from the packs. It takes the packs in
        the order of their element types `packs`, then the inputs of the
        graph in the order of their names `inputs`."""
        lines = self.declarations()
        readonly = "{bufferization.writable = false}"
        if sorted(packs) != sorted(self.packs):
            raise ValueError(f"packs {packs}, not {list(self.packs)}")
        header = [
            f"%pack_{element}: tensor<{self.packs[element]}x{element}> {readonly}"
            for element in packs
        ]
        header += [
            f"%in_{name}: {tensor_type(self.meta[name])} {readonly}" for name in inputs
        ]
        body, outputs = self.launches(
            "",
            {element: f"%pack_{element}" for element in packs},
            {name: f"%in_{name}" for name in inputs},
        )
        returned = [tensor_type(self.meta[name]) for name in self.output.args]
        body.append(f"return {', '.join(outputs)} : {', '.join(returned)}")
        lines.append(
            f"func.func @{function}({', '.join(header)}) -> ({', '.join(returned)}) {{\n  "
            + "\n  ".join(body)
            + "\n}"
        )
        with ir.Context() as context:
            context.allow_unregistered_dialects = True
            return str(ir.Module.parse("\n".join(lines)))

    def module(self, hart_class):
        """The graphs of the classes of `hart_class` as one MLIR module, each
        from its first instance: the imported graph, the private
        <graph>_graph, and <graph>, which takes its parameters (each a slice
        of a pack, reshaped here) and its inputs, and writes its outputs
        into its last arguments (destination passing: an output of a dynamic
        size is a buffer of the program, which knows the sizes). <graph>
        returns them, each the value of its argument, which
        drop-equivalent-buffer-results removes after bufferization."""
        texts = []
        for graph_name, segment in self.graphs.items():
            if segment.hart_class != hart_class:
                continue
            part = Graph(
                dict(self.graph._ops_registry), f"{graph_name}_graph", self.graph.device
            )
            for name in segment.params:
                part.add_node(
                    copied(self.parameters[name][0], name, []), NodeType.FakeNode
                )
            for name in segment.inputs:
                placeholder = PlaceholderOp()
                placeholder.name = name
                placeholder.tensor_meta = dict(self.meta[name])
                part.add_node(placeholder, NodeType.InputNode)
            for _, node in segment.ops:
                node._children = []
                part.add_node(node)
            returned = OutputOp()
            returned.name = "output"
            for name in segment.outputs:
                returned.add_argument(name)
            returned._parents = list(returned.args)
            part.add_node(returned)
            for node in part.body:
                for parent in node.parents:
                    if node.name not in part.node_table[parent]._children:
                        part.node_table[parent]._children.append(node.name)
            part.lower_to_top_level_ir(buffer_layout="contiguous", writable=False)
            for op in part._imported_module.body.operations:
                # A rank owns its state slice. Permit bufferization to update
                # it in place instead of cloning the complete KV cache.
                with op.context:
                    attributes = list(op.attributes["arg_attrs"])
                    for index, name in enumerate(segment.params + segment.inputs):
                        if name in self.slices:
                            entries = {
                                item.name: item.attr for item in attributes[index]
                            }
                            entries["bufferization.writable"] = ir.BoolAttr.get(True)
                            if self.layout(name):
                                entries["bufferization.buffer_layout"] = (
                                    ir.Attribute.parse(self.layout(name))
                                )
                            attributes[index] = ir.DictAttr.get(entries)
                    op.attributes["arg_attrs"] = ir.ArrayAttr.get(attributes)
                op.attributes["sym_visibility"] = ir.StringAttr.get(
                    "private", op.context
                )
                texts.append(str(op))

            arguments, results = self.signature(segment)
            readonly = "{bufferization.writable = false}"
            count = len(segment.params)
            header = [
                f"%param{i}: {t} {readonly}" for i, t in enumerate(arguments[:count])
            ]
            header += [
                f"%input{i}: {t} "
                + (
                    "{bufferization.writable = true"
                    + (
                        f", bufferization.buffer_layout = {self.layout(segment.inputs[i])}"
                        if self.layout(segment.inputs[i])
                        else ""
                    )
                    + "}"
                    if segment.inputs[i] in self.slices
                    else readonly
                )
                for i, t in enumerate(arguments[count:])
            ]
            header += [f"%output{i}: {t}" for i, t in enumerate(results)]
            body = []
            types = []
            for index, name in enumerate(segment.params):
                shape = [int(d) for d in self.parameters[name][0].tensor_meta["shape"]]
                kind = tensor_type(self.parameters[name][0].tensor_meta)
                body.append(
                    f"%shape{index} = arith.constant dense<{shape}> : tensor<{len(shape)}xindex>"
                    if shape
                    else f"%shape{index} = tensor.empty() : tensor<0xindex>"
                )
                body.append(
                    f"%reshaped{index} = tensor.reshape %param{index}(%shape{index})"
                    f" : ({arguments[index]}, tensor<{len(shape)}xindex>) -> {kind}"
                )
                types.append(kind)
            operands = [f"%reshaped{i}" for i in range(count)]
            operands += [f"%input{i}" for i in range(len(segment.inputs))]
            types += arguments[count:]
            body.append(
                f"%results:{len(results)} = func.call @{graph_name}_graph({', '.join(operands)})"
                f" : ({', '.join(types)}) -> ({', '.join(results)})"
            )
            for i, t in enumerate(results):
                body.append(
                    f"%written{i} = bufferization.materialize_in_destination %results#{i} in %output{i}"
                    f" : ({t}, {t}) -> {t}"
                )
            body.append(
                f"return {', '.join(f'%written{i}' for i in range(len(results)))} : {', '.join(results)}"
            )
            texts.append(
                f"func.func @{graph_name}({', '.join(header)}) -> ({', '.join(results)}) {{\n  "
                + "\n  ".join(body)
                + "\n}"
            )
        with ir.Context() as context:
            context.allow_unregistered_dialects = True
            return str(ir.Module.parse("\n".join(texts)))

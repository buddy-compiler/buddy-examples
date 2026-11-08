"""Partition a graph across cores with tensor parallelism, as a device
program.

Core 0 is the reduce core; cores 1..count run ranks 0..count-1. The recipe
chooses the Linears (TransposeMatmulFusedOp) split along their output
channels: their MXINT8 blocks are dealt out to the ranks, the first ranks one
more when they do not divide evenly, and a fused Linear (q, k, v or gate, up
in one) deals each of its segments out on its own. Every value computed from
a sharded value is sharded along one axis, and each rank computes its pieces
of it. A Linear whose input is sharded is split along K: each rank computes a
partial sum, and the reduce core adds them up. The other Linears run whole.
States (the KV cache) are sharded inputs: each rank owns its slice. Sharded
outputs are joined by the reduce core. Weights are MXINT8 packed along K; a
split keeps whole blocks.

The other operations run on the reduce core, except the `replicated` ones
(positions, masks, rotary tables, and the keys and values when there are
fewer KV heads than ranks): every segment that uses them computes them
itself, and reads its pieces of those that meet a sharded value along its
sharded axis: the reduce core deals its pieces of a value it computes out to
the ranks; a parameter is sliced offline; a value the same at every offset of
the axis (a mask broadcast over the heads) is read from the start by every
rank alike.

The result is a device program (Device): each core's operations cut into
segments where they read a value another core computes, and the outputs the
reduce core computes (the joins of the sharded ones, the replicated ones)
apart. The recipe names the semantic graph class of every segment (the
attention of a rank, its MLP, the end of a block, ...): the segments of a
class are instances of one graph, defined once with its parameters as
arguments and launched on a class of core, `reduce` or `poly`. Which core
runs a launch and how a value moves between cores are the device
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
    AddOp,
    AmaxOp,
    BatchMatmulOp,
    CatOp,
    ExpandOp,
    FullOp,
    GetItemOp,
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
    TransposeMatmulFusedOp,
    UnbindOp,
    UnsqueezeOp,
    ViewOp,
)
from buddy.compiler.ops.utils import mlir_element_type_get

from .quant import FP32ToMXINT8QuantOp, MXINT8ToFP32DequantOp


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
        if axis != 0:
            raise ValueError(f"{node.name}: sharded along {axis}, not the batch")
        return 0, held
    if isinstance(node, (SliceOp, CatOp)) and node.args[1] % rank == axis:
        raise ValueError(f"{node.name}: slices or joins the sharded axis")
    if isinstance(node, (MeanOp, SumDimOp, AmaxOp)):
        reduced = [d % rank for d in node.args[1]]
        if axis in reduced:
            raise ValueError(f"{node.name}: reduces the sharded axis")
        keep = len(node.args) > 2 and node.args[2]
        return axis - (0 if keep else sum(d < axis for d in reduced)), held
    # Elementwise: the arguments broadcast from the right.
    return axis + len(shape) - rank, held


def spanned(graph, node, axes):
    """The unsharded arguments of the sharded `node` that span its sharded
    axis whole, with their axis along it: each rank reads its pieces."""
    shape = node.tensor_meta["shape"]
    result = {}
    for name in node.parents:
        if name in axes:
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
    if isinstance(node, (FP32ToMXINT8QuantOp, MXINT8ToFP32DequantOp)):
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


def partition(graph, params, count, states, replicated, columns, segments, classify):
    """The device program of the reduce core and the `count` ranks.
    states: the sharded inputs and their sharded axis. replicated: the
    operations every core computes for itself. columns: the
    Linears split along their output channels. segments: the fused Linears
    among them (q, k, v or gate, up in one) and their number of segments,
    each split across the ranks on its own. classify(device, segment): the
    name of the graph class of a segment."""
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
    axes, pieces = {}, {}
    for name, axis in states.items():
        n = graph.node_table[name].tensor_meta["shape"][axis]
        if n % count:
            raise ValueError(f"{name}: {n} does not split into {count}")
        axes[name] = axis
        pieces[name] = [[(r * n // count, (r + 1) * n // count)] for r in range(count)]
    rows = set()
    for node in body:
        if node.name in replicated:
            continue
        if isinstance(node, TransposeMatmulFusedOp) and node.args[0] in axes:
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
            if segment % 32:
                raise ValueError(
                    f"{node.name}: {segment} channels are not MXINT8 blocks"
                )
            base, extra = divmod(segment // 32, count)
            held = []
            for r in range(count):
                low = 32 * (r * base + min(r, extra))
                high = low + 32 * (base + (r < extra))
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
            if child not in axes and child not in rows and child != output.name:
                raise ValueError(f"{child} uses the sharded {name} whole")
    # A projection that is not sharded still runs on a compute core.
    whole = {
        node.name
        for node in body
        if isinstance(node, TransposeMatmulFusedOp)
        and node.name not in replicated | columns | rows
    }
    weights = {graph.node_table[name].args[1] for name in columns | rows | whole}
    for name in weights:
        restored = graph.node_table[name]
        if not isinstance(restored, MXINT8ToFP32DequantOp) or restored.args[1] != 1:
            raise ValueError(f"{name}: a split weight is not MXINT8 along K")
    restored_inputs = {
        node.name
        for node in body
        if isinstance(node, MXINT8ToFP32DequantOp)
        and all(child in columns | whole for child in node.children)
    }

    def local(node, rank):
        """The shape of rank `rank`'s piece of the sharded `node`."""
        shape = list(node.tensor_meta["shape"])
        width = total(pieces[node.name][rank])
        if (
            isinstance(node, FP32ToMXINT8QuantOp)
            and node.args[1] % len(shape) == axes[node.name]
        ):
            width += width // 32
        shape[axes[node.name]] = width
        return shape

    # ===------------------------------------------------------------------===
    # The operations of each core, each keyed by the graph position of the
    # operation it comes from, with the sliced weights and states as their
    # parameters and inputs.
    # ===------------------------------------------------------------------===
    operations = [[] for _ in range(count + 1)]
    placeholders = [[] for _ in range(count + 1)]
    local_values = [{} for _ in range(count + 1)]
    position = {node.name: index for index, node in enumerate(graph.body)}

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
            or node.name in restored_inputs
            or node.name in replicated
        ):
            continue
        if node.name in whole:
            arguments = list(node.args)
            for index in (0, 1):
                source = graph.node_table[node.args[index]]
                if isinstance(source, MXINT8ToFP32DequantOp):
                    name = f"{source.name}_{node.name}_whole"
                    operations[1].append((key, copied(source, name, list(source.args))))
                    arguments[index] = name
            operations[1].append((key, copied(node, node.name, arguments)))
        elif node.name in columns or node.name in rows:
            restored = graph.node_table[node.args[1]]
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
                        f"{restored.name}_r{rank}",
                        *node.args[2:],
                    ]
                    if source.name in restored_inputs:
                        arguments[0] = f"{source.name}_r{rank}"
                        operations[rank + 1].append(
                            (key, copied(source, arguments[0], list(source.args)))
                        )
                    result = local(node, rank)
                else:
                    ((low, high),) = pieces[node.args[0]][rank]
                    if low % 32 or high % 32:
                        raise ValueError(
                            f"{node.name}: rank {rank} K {low}..{high} splits MXINT8 blocks"
                        )
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
                # With one rank the product of a row Linear is its whole sum.
                name = (
                    node.name
                    if node.name in rows and count == 1
                    else f"{node.name}_r{rank}"
                )
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
                    # A value of the reduce core is sliced there: each rank
                    # receives its pieces only. A value the same at every
                    # offset along the axis (a mask broadcast over the
                    # heads): each rank reads its pieces from the start, as
                    # every other rank.
                    core = rank + 1
                    if name not in replicated and not isinstance(source, PlaceholderOp):
                        core = 0
                    same = uniform(graph, name, axis)
                    parts = []
                    for index, (low, high) in enumerate(pieces[node.name][rank]):
                        if same:
                            low, high = 0, high - low
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
                        operations[core].append((key, piece))
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
                        operations[core].append((key, joined))
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
    # The outputs: core 0 joins the slices the ranks hold of every sharded
    # one. A replicated output is materialized once on the first compute
    # core; its consumers share that value instead of recomputing it.
    # ===------------------------------------------------------------------===
    joins = set()
    for name in output.args:
        if name in replicated:
            node = graph.node_table[name]
            replicated.remove(name)
            operations[1].append((position[name], copied(node, name, list(node.args))))
            joins.add(name)
            continue
        if name not in axes:
            continue
        held = pieces[name]
        if any(len(rank) != 1 for rank in held) or [rank[0][0] for rank in held] != [
            0
        ] + [rank[0][1] for rank in held[:-1]]:
            raise ValueError(
                f"{name}: the slices of the ranks do not join in rank order"
            )
        joined = CatOp()
        joined.name = name
        parts = [f"{name}_r{rank}" for rank in range(count)]
        joined.add_argument(parts)
        joined.add_argument(axes[name])
        for part in parts:
            joined.add_parent(part)
        meta = graph.node_table[name].tensor_meta
        joined.tensor_meta = {"shape": list(meta["shape"]), "dtype": meta["dtype"]}
        operations[0].append((len(graph.body), joined))
        joins.add(name)

    parameters = {node.name: (node, values[node.name]) for node in graph.params}
    inputs = {node.name: node for node in graph.inputs}
    slices = {}
    for core in range(1, count + 1):
        for node, kind in placeholders[core]:
            if kind == NodeType.FakeNode:
                parameters[node.name] = (node, local_values[core][node.name])
            else:
                state = node.name.rsplit("_r", 1)[0]
                ((low, high),) = pieces[state][core - 1]
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
    )


class Segment:
    """Operations of one core between two receptions, or the outputs the
    reduce core computes there (`join`): an instance of `graph`, the graph of its class
    `graph_class`, which the device program launches on a core of
    `core_class`."""

    def __init__(self, core, core_class, join):
        self.core = core
        self.core_class = core_class
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
    """The partitioned graph as a device program: the operations of each core
    cut into segments where they receive a value of another core, each
    segment a graph launched on the class of its core — `reduce` for core 0,
    `poly` for the ranks — and the program that launches them in an order
    their values allow, on the reduce core. Placement on the cores and the
    transfers between them are the device compiler's.

    Every segment computes the replicated operations it uses itself. The
    program takes the parameters packed into one buffer per element type,
    then the inputs of the graph; it hands each rank its slice of every state
    and returns the outputs of the graph, which core 0 computes or joins."""

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
    ):
        self.graph = graph
        self.ranks = ranks
        self.parameters = parameters
        self.inputs = inputs
        self.slices = slices
        self.output = output
        producer = {
            node.name: core for core, ops in enumerate(operations) for _, node in ops
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
        # it. The operations of a core at one depth form a segment: a value
        # of another core comes from a smaller depth, so the segments depend
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
        # weight) joins the segment of its first reader on its core.
        readers = {}
        for name, node in nodes.items():
            for parent in node.parents:
                readers.setdefault(parent, []).append(name)
        for core, ops in enumerate(operations):
            for _, node in reversed(ops):
                if not reads(node.parents):
                    depth[node.name] = min(
                        (
                            depth[r]
                            for r in readers.get(node.name, [])
                            if producer[r] == core
                        ),
                        default=0,
                    )
        segments = []
        for core, ops in enumerate(operations):
            levels = {}
            for key, node in ops:
                level = (depth[node.name], node.name in joins)
                if level not in levels:
                    levels[level] = Segment(
                        core, "poly" if core else "reduce", level[1]
                    )
                    segments.append(levels[level])
                levels[level].ops.append((key, node))
        owner = {node.name: segment for segment in segments for _, node in segment.ops}

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
                raise ValueError("the segments depend on each other in a cycle")
            chosen = min(ready, key=lambda s: (first[id(s)], s.core))
            del waiting[id(chosen)]
            for dependencies in waiting.values():
                dependencies.discard(id(chosen))
            self.segments.append(chosen)
        counts = {}
        for segment in self.segments:
            index = counts.get(segment.core, 0)
            counts[segment.core] = index + 1
            segment.name = (
                f"reduce{index}"
                if segment.core == 0
                else f"rank{segment.core - 1}_{index}"
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
        """What a segment computes, up to the names of its values: its core
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
            segment.core_class,
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

    def signature(self, segment):
        """The argument and result types of the graph of a segment: its
        parameters, each as a slice of a pack, its inputs; its outputs."""
        arguments = [self.flat(name) for name in segment.params]
        arguments += [tensor_type(self.meta[name]) for name in segment.inputs]
        return arguments, [tensor_type(self.meta[name]) for name in segment.outputs]

    def program(self):
        """The device program, `program`, as MLIR: a launch of the graph of
        every segment, its parameters sliced from the packs."""
        lines = []
        for name, segment in self.graphs.items():
            arguments, results = self.signature(segment)
            lines.append(
                f"func.func private @{name}({', '.join(arguments)}) -> ({', '.join(results)})"
            )
        readonly = "{bufferization.writable = false}"
        header = [
            f"%pack_{element}: tensor<{size}x{element}> {readonly}"
            for element, size in self.packs.items()
        ]
        graph_inputs = [node.name for node in self.graph.inputs]
        header += [
            f"%in_{name}: {tensor_type(self.meta[name])} {readonly}"
            for name in graph_inputs
        ]
        body = []
        values = {name: f"%in_{name}" for name in graph_inputs}
        # Each symbolic size, from an input that has it as a size; a size
        # that is an expression of them (the KV length after the tokens),
        # computed with arith.
        sizes = {}
        for name in graph_inputs:
            for axis, dimension in enumerate(self.meta[name]["shape"]):
                if (
                    isinstance(dimension, torch.SymInt)
                    and dimension.node.expr.is_Symbol
                    and dimension.node.expr not in sizes
                ):
                    body.append(f"%axis_{name}_{axis} = arith.constant {axis} : index")
                    body.append(
                        f"%size_{dimension} = tensor.dim %in_{name}, %axis_{name}_{axis} : {tensor_type(self.meta[name])}"
                    )
                    sizes[dimension.node.expr] = f"%size_{dimension}"

        def size(expression):
            if expression not in sizes:
                name = f"%size{len(sizes)}"
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
                f"%state_{name} = tensor.extract_slice %in_{state}[{', '.join(map(str, offsets))}]"
                f" [{', '.join(map(str, extent))}] [{', '.join('1' for _ in shape)}]"
                f" : {tensor_type(self.meta[state])} to {tensor_type(self.meta[name])}"
            )
            values[name] = f"%state_{name}"
        # Contiguous output shards can be produced directly in disjoint
        # slices of the final tensor, without a gather task on the control core.
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
                joined[node.name] = f"%joined_{node.name}"
                body.append(
                    f"{joined[node.name]} = tensor.empty({operands})"
                    f" : {tensor_type(node.tensor_meta)}"
                )
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
                    f"%param_{name} = tensor.extract_slice %pack_{element}[{offset}] [{count}] [1]"
                    f" : tensor<{self.packs[element]}x{element}> to tensor<{count}x{element}>"
                )
                values[name] = f"%param_{name}"
            arguments, results = self.signature(segment)
            outs = []
            for index, name in enumerate(segment.outputs):
                operands = ", ".join(
                    size(d.node.expr)
                    for d in self.meta[name]["shape"]
                    if isinstance(d, torch.SymInt)
                )
                if name in destinations:
                    whole, region = destinations[name]
                    body.append(
                        f"%{segment.name}_out{index} = tensor.extract_slice"
                        f" {joined[whole]}{region} : {tensor_type(self.meta[whole])}"
                        f" to {results[index]}"
                    )
                else:
                    body.append(
                        f"%{segment.name}_out{index} = tensor.empty({operands}) : {results[index]}"
                    )
                outs.append(f"%{segment.name}_out{index}")
            many = f":{len(results)}" if len(results) > 1 else ""
            operands = [values[name] for name in segment.params + segment.inputs] + outs
            body.append(
                f'%{segment.name}{many} = "device.launch"({", ".join(operands)})'
                f' <{{callee = @{segment.graph}, core_class = "{segment.core_class}",'
                f" operandSegmentSizes = array<i32: {len(arguments)}, {len(results)}>}}>"
                f" {{device.rank = {segment.core - 1 if segment.core else 0} : i64}}"
                f" : ({', '.join(arguments + results)}) -> ({', '.join(results)})"
            )
            for index, name in enumerate(segment.outputs):
                values[name] = (
                    f"%{segment.name}#{index}" if many else f"%{segment.name}"
                )
                if name in destinations:
                    whole, region = destinations[name]
                    result = f"%joined_{name}"
                    body.append(
                        f"{result} = tensor.insert_slice {values[name]} into"
                        f" {joined[whole]}{region} : {results[index]}"
                        f" into {tensor_type(self.meta[whole])}"
                    )
                    joined[whole] = result
        returned = [tensor_type(self.meta[name]) for name in self.output.args]
        body.append(
            f"return {', '.join(values[name] for name in self.output.args)} : {', '.join(returned)}"
        )
        lines.append(
            f"func.func @program({', '.join(header)}) -> ({', '.join(returned)}) {{\n  "
            + "\n  ".join(body)
            + "\n}"
        )
        with ir.Context() as context:
            context.allow_unregistered_dialects = True
            return str(ir.Module.parse("\n".join(lines)))

    def module(self, core_class):
        """The graphs of the classes of `core_class` as one MLIR module, each
        from its first instance: the imported graph, the private
        <graph>_graph, and <graph>, which takes its parameters (each a slice
        of a pack, reshaped here) and its inputs, and writes its outputs
        into its last arguments (destination passing: an output of a dynamic
        size is a buffer of the program, which knows the sizes). <graph>
        returns them, each the value of its argument, which
        drop-equivalent-buffer-results removes after bufferization."""
        texts = []
        for graph_name, segment in self.graphs.items():
            if segment.core_class != core_class:
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
                f"%input{i}: {t} {readonly}" for i, t in enumerate(arguments[count:])
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

"""Explicit persistent V and packed K-head/K-tail state for Qwen3-8B."""

import torch
import buddy_mlir.ir as ir
from buddy_mlir.dialects import arith, tensor
from buddy.compiler.graph import NodeType
from buddy.compiler.graph.operation import (
    AddOp,
    IndexPutOp,
    IotaOp,
    Op,
    OpType,
    OutputOp,
    PlaceholderOp,
    SubOp,
)
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from buddy.compiler.graph.type import TensorDType
from compiler.frontend.quant import (
    FP32ToMXINT8QuantOp,
    MXINT8ToFP32DequantOp,
    quantize_mxint8,
    dequantize_mxint8,
)
from frontend.quant.graph import operation, replace_argument


class PackedKUpdateOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.SliceLikeType


def update(node, symbols):
    cache, position, values = [symbols[(str(name), 0)] for name in node.args]
    shape = list(ir.RankedTensorType(values.type).shape)
    index = ir.IndexType.get()
    # The position is an input. Materialize it before asynchronous state writers.
    with ir.InsertionPoint.at_block_begin(ir.InsertionPoint.current.block):
        zero = arith.ConstantOp(index, ir.IntegerAttr.get(index, 0)).result
        start = arith.IndexCastOp(index, tensor.ExtractOp(position, [zero])).result
    sizes = []
    for axis, size in enumerate(shape):
        if size < 0:
            dim = arith.ConstantOp(index, ir.IntegerAttr.get(index, axis)).result
            sizes.append(tensor.DimOp(values, dim).result)
    return tensor.InsertSliceOp(
        values,
        cache,
        [start],
        sizes,
        [],
        ir.DenseI64ArrayAttr.get([0, 0, ir.ShapedType.get_dynamic_size(), 0]),
        ir.DenseI64ArrayAttr.get(shape),
        ir.DenseI64ArrayAttr.get([1, 1, 1, 1]),
    ).result


def packed_keys(keys, values):
    head = [quantize_mxint8(key, 3) for key in keys]
    tail = [
        quantize_mxint8(key - dequantize_mxint8(code, 3), 3)
        for key, code in zip(keys, head, strict=True)
    ]
    return [*values, *head, *tail]


def persistent_keys(graph, order):
    """Replace each unused FP32 K plane with its two exact packed planes."""
    original = dict(zip(order, graph.inputs, strict=True))
    if len(original) != 74 or set(original) != set(range(74)):
        raise ValueError("Expected Qwen3-8B tokens, position, 36 K and 36 V inputs")
    (output,) = [node for node in graph.body if isinstance(node, OutputOp)]
    if len(output.args) != 73:
        raise ValueError("Expected logits, 36 K and 36 V outputs")
    semantic = {original[0].name: 0, original[1].name: 1}
    heads, tails, layout = [], [], []
    for layer in range(36):
        old = original[2 + layer]
        (name,) = old.children
        inserted = graph.node_table[name]
        if not isinstance(inserted, IndexPutOp) or inserted.args[0] != old.name:
            raise ValueError(f"{old.name}: expected one contiguous K-cache update")
        positions = graph.node_table[inserted.args[1][2]]
        if not isinstance(positions, AddOp) or positions.args[0] != original[1].name:
            raise ValueError(f"{name}: expected input position plus token iota")
        iota = graph.node_table[positions.args[1]]
        if (
            not isinstance(iota, IotaOp)
            or iota.kwargs["start"] != 0
            or iota.kwargs["step"] != 1
        ):
            raise ValueError(f"{name}: expected contiguous unit-step token positions")
        source = graph.node_table[inserted.args[2]]
        shape = list(old.tensor_meta["shape"])
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 8 or shape[3] != 128:
            raise ValueError(f"{name}: expected [1,8,capacity,128] K-cache")
        readers = [user for user in inserted.children if user != output.name]
        quantizers = [
            graph.node_table[user]
            for user in readers
            if isinstance(graph.node_table[user], FP32ToMXINT8QuantOp)
        ]
        residuals = [
            graph.node_table[user]
            for user in readers
            if isinstance(graph.node_table[user], SubOp)
        ]
        if len(quantizers) != 1 or len(residuals) != 1 or len(readers) != 2:
            raise ValueError(f"{name}: unexpected FP32 K consumers {readers}")
        head, residual = quantizers[0], residuals[0]
        decoded = graph.node_table[residual.args[1]]
        (tail_name,) = residual.children
        tail = graph.node_table[tail_name]
        if (
            not isinstance(decoded, MXINT8ToFP32DequantOp)
            or decoded.args != [head.name, 3]
            or head.args[1] != 3
            or not isinstance(tail, FP32ToMXINT8QuantOp)
            or tail.args[1] != 3
        ):
            raise ValueError(
                f"{name}: expected ordered head/decode/subtract/tail quantization"
            )
        newshape = list(source.tensor_meta["shape"])
        if len(newshape) != 4 or newshape[:2] != [1, 8] or newshape[3] != 128:
            raise ValueError(f"{name}: expected contiguous new K token rows")
        if iota.tensor_meta["shape"][0] != newshape[2]:
            raise ValueError(f"{name}: position count differs from new K token count")
        packed = [*shape[:-1], 132]
        states = []
        for suffix, semantic_index in (("head", 38 + layer), ("tail", 74 + layer)):
            state = PlaceholderOp()
            state.name = f"{old.name}_{suffix}_state"
            state.tensor_meta = {"shape": list(packed), "dtype": TensorDType.Int8}
            graph.add_node(state, NodeType.InputNode)
            semantic[state.name] = semantic_index
            states.append(state)
        replace_argument(graph, head, 0, source)
        head.tensor_meta["shape"] = [*newshape[:-1], 132]
        local_decoded = operation(
            graph,
            MXINT8ToFP32DequantOp,
            head.name + "_new_values",
            [head],
            [3],
            newshape,
            TensorDType.Float32,
        )
        replace_argument(graph, residual, 0, source)
        replace_argument(graph, residual, 1, local_decoded)
        residual.tensor_meta["shape"] = list(newshape)
        tail.tensor_meta["shape"] = [*newshape[:-1], 132]
        writes = []
        for state, quantized in zip(states, (head, tail), strict=True):
            write = operation(
                graph,
                PackedKUpdateOp,
                state.name + "_updated",
                [state, original[1], quantized],
                [],
                packed,
                TensorDType.Int8,
            )
            writes.append(write)
        replace_argument(graph, decoded, 0, writes[0])
        for user in list(tail.children):
            if user != writes[1].name:
                replace_argument(graph, graph.node_table[user], 0, writes[1])
        if output.args[1 + layer] != inserted.name:
            raise ValueError(f"{name}: K output order differs from its layer")
        for write in writes:
            write.add_children(output.name)
        heads.append(writes[0].name)
        tails.append(writes[1].name)
        graph.delete_node(
            inserted, [graph.node_table[parent] for parent in inserted.parents]
        )
        graph.delete_node(old, [])
    for layer in range(36):
        semantic[original[38 + layer].name] = 2 + layer
        layout.append((8, 512, 512, 0))
    layout.extend([(8, 132, 128, 127)] * 72)
    output.args[:] = [output.args[0], *output.args[37:], *heads, *tails]
    sort_graph(graph)
    return [semantic[node.name] for node in graph.inputs], layout

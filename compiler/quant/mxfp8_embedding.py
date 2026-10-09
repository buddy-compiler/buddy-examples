import math

import torch
from buddy.compiler.graph.operation import EmbeddingOp, Op, OpType
from buddy.compiler.graph.type import TensorDType
from buddy.compiler.graph.transform.quantization.mxfp8 import quantize
from buddy.compiler.ops.tosa import _create_shape_operand
from buddy_mlir import ir
from buddy_mlir.dialects import tosa

from .mxfp8_lowering import lower_codec


class MXFP8EmbeddingOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ReshapeType


def lower_embedding(node, symbols):
    weight, indices = [symbols[(name, 0)] for name in node.args[:2]]
    weight_type, index_type = ir.RankedTensorType(weight.type), ir.RankedTensorType(
        indices.type
    )
    index_shape = list(index_type.shape)
    shape = list(node.tensor_meta["shape"])
    width = shape[-1]
    if (
        len(weight_type.shape) != 2
        or len(index_shape) not in (1, 2)
        or any(size <= 0 for size in [*weight_type.shape, *index_shape])
        or width <= 0
        or width % 32
        or list(weight_type.shape)[1] != width + width // 32
        or weight_type.element_type != ir.IntegerType.get_signless(8)
        or index_type.element_type
        not in (ir.IntegerType.get_signless(32), ir.IntegerType.get_signless(64))
        or shape != [*index_shape, width]
        or node.tensor_meta["dtype"] != TensorDType.Float32
    ):
        raise ValueError(
            "MXFP8 embedding requires row-packed INT8 weights and static integer indices"
        )
    count = math.prod(index_shape)
    ids = tosa.ReshapeOp(indices, _create_shape_operand([1, count])).result
    if index_type.element_type != ir.IntegerType.get_signless(32):
        ids = tosa.CastOp(
            ir.RankedTensorType.get([1, count], ir.IntegerType.get_signless(32)), ids
        ).result
    row_stride = width + width // 32
    rows = tosa.ReshapeOp(weight, _create_shape_operand([1, *weight_type.shape])).result
    gathered = tosa.GatherOp(
        ir.RankedTensorType.get([1, count, row_stride], weight_type.element_type),
        rows,
        ids,
    ).output
    codes = tosa.SliceOp(
        ir.RankedTensorType.get([1, count, width], weight_type.element_type),
        gathered,
        _create_shape_operand([0, 0, 0]),
        _create_shape_operand([1, count, width]),
    ).result
    scales = tosa.SliceOp(
        ir.RankedTensorType.get([1, count, width // 32], weight_type.element_type),
        gathered,
        _create_shape_operand([0, 0, width]),
        _create_shape_operand([1, count, width // 32]),
    ).result
    codes = tosa.ReshapeOp(codes, _create_shape_operand([*index_shape, width])).result
    scales = tosa.ReshapeOp(
        scales, _create_shape_operand([*index_shape, width // 32])
    ).result
    return lower_codec([codes, scales], encoding=False)


def pack_rows(values):
    if (
        not isinstance(values, torch.Tensor)
        or values.dtype != torch.float32
        or values.ndim != 2
        or values.shape[0] <= 0
        or values.shape[1] <= 0
        or values.shape[1] % 32
    ):
        raise ValueError(
            "MXFP8 embedding requires a positive FP32 matrix with width divisible by 32"
        )
    rows, width = values.shape
    packed = torch.empty(
        (rows, width + width // 32), dtype=torch.int8, device=values.device
    )
    chunk_rows = max(1, (1 << 24) // width)
    for first in range(0, rows, chunk_rows):
        codes, scales = quantize(values[first : first + chunk_rows])
        count = codes.shape[0]
        packed[first : first + count, :width] = codes
        packed[first : first + count, width:] = scales
    return packed


def _quantize_embeddings(graph, params, names, *, embeddings):
    parameters = list(graph.params)
    positions = {node.name: index for index, node in enumerate(parameters)}
    nodes = []
    uses = {}
    for node in graph.body:
        if not isinstance(node, EmbeddingOp):
            continue
        weight_name = node.args[0]
        if (
            weight_name not in positions
            or names[positions[weight_name]] not in embeddings
        ):
            continue
        nodes.append(node)
        uses.setdefault(weight_name, set()).add(node.name)
    if {names[positions[name]] for name in uses} != embeddings:
        raise ValueError(
            "Selected embeddings do not match the graph embedding operations"
        )
    quantized = {}
    for weight_name, consumers in uses.items():
        weight = graph.node_table[weight_name]
        if set(weight._children) != consumers:
            raise ValueError(
                f"Embedding parameter requires a separate packed view for other consumers: {weight_name}"
            )
        index = positions[weight_name]
        value = params[index].detach()
        if value.ndim != 2 or value.shape[0] <= 0:
            raise ValueError("MXFP8 embedding requires a positive FP32 matrix")
        packed = pack_rows(value)
        params[index] = packed
        weight.tensor_meta["shape"] = list(packed.shape)
        weight.tensor_meta["dtype"] = TensorDType.Int8
        quantized[weight_name] = (
            list(value.shape),
            packed,
            {"row_stride": packed.shape[1]},
        )
    for node in nodes:
        replacement = MXFP8EmbeddingOp()
        replacement._name = node.name
        replacement._arguments = list(node.args)
        replacement._parents = list(node._parents)
        replacement._children = list(node._children)
        replacement._tensor_meta = dict(node.tensor_meta)
        graph._body[graph._body.index(node)] = replacement
        graph.node_table[node.name] = replacement
    for group in graph.op_groups:
        graph.op_groups[group] = [
            graph.node_table[node.name] for node in graph.op_groups[group]
        ]
    graph._ops_registry["MXFP8EmbeddingOp"] = lower_embedding
    return quantized


def quant_embedding(graph, params, names, output, name, packer, *, embeddings):
    from .mxfp8_graph import _write_parameters

    quantized = _quantize_embeddings(graph, params, names, embeddings=embeddings)
    return _write_parameters(graph, params, names, output, name, packer, quantized)

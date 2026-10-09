import torch
from buddy.compiler.graph.operation import MatmulOp, AddMMOp, TOp, PermuteOp, Op, OpType
from buddy.compiler.graph.type import TensorDType
from buddy_mlir import ir
from buddy.compiler.graph.transform.quantization.mxfp8 import quantize
from .rax import QuantTensor, RaxQuantPackage, write_rax
from examples.balls.mxmm.compiler.python.layout import plan, pack


class MXFP8MatmulOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ReduceType


def lower_matmul(node, symbols):
    activation, weight = [symbols[(name, 0)] for name in node.args]
    shape = list(ir.RankedTensorType(activation.type).shape)
    if len(shape) != 2 or any(size <= 0 for size in shape) or shape[1] % 32:
        raise ValueError(
            "MXFP8 requires a positive static FP32 matrix with K divisible by 32"
        )
    rows, reduction = shape
    tile_rows = 1 if rows == 1 else min(node.layout["tile_m"], (rows + 15) // 16 * 16)
    chunk, stride = node.layout["tile_k"], node.layout["bank_bytes"]
    activation_chunk = (
        reduction
        if reduction <= 65535
        and (tile_rows * reduction * 33 // 32 + 15) // 16 * 16 <= stride
        else chunk
    )
    packed_size = (
        (rows + tile_rows - 1)
        // tile_rows
        * ((reduction + activation_chunk - 1) // activation_chunk)
        * stride
    )
    integer = ir.IntegerType.get_signless(64)
    packed = ir.Operation.create(
        "buckyball.mxfp8_quant",
        operands=[activation],
        results=[
            ir.RankedTensorType.get([packed_size], ir.IntegerType.get_signless(8))
        ],
        attributes={
            "tile_rows": ir.IntegerAttr.get(integer, tile_rows),
            "tile_k": ir.IntegerAttr.get(integer, activation_chunk),
            "bank_bytes": ir.IntegerAttr.get(integer, stride),
        },
    ).result
    output = ir.RankedTensorType.get(node.tensor_meta["shape"], ir.F32Type.get())
    attributes = {
        key: ir.IntegerAttr.get(integer, node.layout[key])
        for key in ("tile_m", "tile_n", "tile_k", "bank_bytes")
    }
    attributes["tile_m"] = ir.IntegerAttr.get(integer, tile_rows)
    attributes["reduction_k"] = ir.IntegerAttr.get(integer, reduction)
    attributes["activation_tile_k"] = ir.IntegerAttr.get(integer, activation_chunk)
    return ir.Operation.create(
        "buckyball.mxfp8_matmul",
        results=[output],
        attributes=attributes,
        operands=[packed, weight],
    ).result


def quantize_graph(
    graph,
    params,
    names,
    output,
    name,
    packer,
    *,
    weights,
    embeddings,
    bank_bytes,
    rows_hint,
):
    from .mxfp8_embedding import _quantize_embeddings

    if weights & embeddings:
        raise ValueError(
            "Linear and embedding consumers require separate packed parameters"
        )
    parameters = list(graph.params)
    parameter_inputs = [graph._body[i] for i in graph._inputs]
    positions = {node.name: index for index, node in enumerate(parameters)}
    selected_weights = weights
    removed = set()
    quantized = {}
    for node in list(graph._body):
        if isinstance(node, AddMMOp):
            raise ValueError("MXFP8 linear does not support a fused bias")
        if not isinstance(node, MatmulOp):
            continue
        activation, weight_name = node.args
        transpose = graph.node_table[weight_name]
        if not (
            isinstance(transpose, TOp)
            or isinstance(transpose, PermuteOp)
            and list(transpose.args[1]) == [1, 0]
        ):
            raise ValueError(
                f"MXFP8 linear requires a transposed parameter: {node.name}"
            )
        weight = graph.node_table[transpose._parents[0]]
        index = positions[weight.name]
        if names[index] not in selected_weights:
            raise ValueError(
                f"Linear weight has no quantization decision: {names[index]}"
            )
        if weight.name not in quantized:
            array = params[index].detach()
            codes, scales = quantize(array)
            layout = plan(max(16, rows_hint), *array.shape, bank_bytes, mxfp8=True)
            packed = pack(codes, scales, layout)
            quantized[weight.name] = (list(array.shape), packed, layout)
            params[index] = packed
            weight.tensor_meta["shape"] = [packed.numel()]
            weight.tensor_meta["dtype"] = TensorDType.Int8
        replacement = MXFP8MatmulOp()
        replacement.layout = quantized[weight.name][2]
        replacement._name = node.name
        replacement._arguments = [activation, weight.name]
        replacement._parents = [activation, weight.name]
        replacement._children = list(node._children)
        replacement._tensor_meta = dict(node.tensor_meta)
        graph._body[graph._body.index(node)] = replacement
        graph.node_table[node.name] = replacement
        weight._children.append(node.name)
        removed.add(transpose.name)
    for transpose_name in removed:
        transpose = graph.node_table[transpose_name]
        if any(
            not isinstance(graph.node_table[child], MXFP8MatmulOp)
            for child in transpose._children
        ):
            raise ValueError(
                f"MXFP8 weight transpose has another consumer: {transpose_name}"
            )
        graph.node_table[transpose._parents[0]]._children.remove(transpose_name)
        graph._body.remove(transpose)
        del graph.node_table[transpose_name]
    graph._fake_params = [graph._body.index(node) for node in parameters]
    graph._inputs = [graph._body.index(node) for node in parameter_inputs]
    for group in graph.op_groups:
        graph.op_groups[group] = [
            graph.node_table[node.name]
            for node in graph.op_groups[group]
            if node.name not in removed
        ]
    graph._ops_registry["MXFP8MatmulOp"] = lower_matmul
    if {names[positions[node]] for node in quantized} != selected_weights:
        raise ValueError("Selected weights do not match the graph linear operations")
    quantized.update(_quantize_embeddings(graph, params, names, embeddings=embeddings))
    return _write_parameters(graph, params, names, output, name, packer, quantized)


def _write_parameters(graph, params, names, output, name, packer, quantized):
    parameters = list(graph.params)
    positions = {node.name: index for index, node in enumerate(parameters)}
    tensors, weights, floats = [], [], []
    weight_offset = float_offset = 0
    for node, param, parameter_name in zip(parameters, params, names):
        if node.name in quantized:
            shape, packed, layout = quantized[node.name]
            raw = packed.cpu().numpy().tobytes()
            tensors.append(
                QuantTensor(
                    parameter_name,
                    shape,
                    list(packed.shape),
                    "mxfp8",
                    [1],
                    weight_offset,
                    len(raw),
                    0,
                    0,
                    layout,
                )
            )
            weights.append(raw)
            weight_offset += len(raw)
        else:
            if param.dtype != torch.float32:
                raise ValueError(
                    f"Unquantized parameter requires FP32 storage: {parameter_name}"
                )
            raw = param.detach().cpu().contiguous().numpy().tobytes()
            shape = list(param.shape)
            tensors.append(
                QuantTensor(
                    parameter_name,
                    shape,
                    shape,
                    "f32",
                    [],
                    float_offset,
                    len(raw),
                    0,
                    0,
                )
            )
            floats.append(raw)
            float_offset += len(raw)
    package = RaxQuantPackage(tensors, b"".join(weights), b"".join(floats), b"", {})
    output.mkdir(parents=True, exist_ok=True)
    write_rax(package, output / f"{name}.rax", packer, name)
    (output / "weights.bin").write_bytes(package.weights)
    (output / "params.f32").write_bytes(package.params_f32)

    return {names[positions[node]]: value[2] for node, value in quantized.items()}

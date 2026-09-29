import numpy as np
import torch
from buddy.compiler.graph.operation import MatmulOp, AddMMOp, TOp, PermuteOp, Op, OpType
from buddy.compiler.graph.type import TensorDType
from buddy_mlir import ir
from buddy.compiler.graph.transform.quantization.mxfp8 import quantize
from .rax import QuantTensor, RaxQuantPackage, write_rax


class MXFP8MatmulOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ReduceType


def lower_matmul(node, symbols):
    output = ir.RankedTensorType.get(node.tensor_meta["shape"], ir.F32Type.get())
    return ir.Operation.create("buckyball.mxfp8_matmul", results=[output],
                               operands=[symbols[(name, 0)] for name in node.args]).result


def quantize_graph(graph, params, names, output, name, packer, *, weights, reorder, tile_rows, tile_k):
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
        if not (isinstance(transpose, TOp) or
                isinstance(transpose, PermuteOp) and list(transpose.args[1]) == [1, 0]):
            raise ValueError(f"MXFP8 linear requires a transposed parameter: {node.name}")
        weight = graph.node_table[transpose._parents[0]]
        index = positions[weight.name]
        if names[index] not in selected_weights:
            raise ValueError(f"Linear weight has no quantization decision: {names[index]}")
        if weight.name not in quantized:
            array = params[index].detach().numpy()
            codes, scales = quantize(array)
            packed = reorder(codes, scales)
            quantized[weight.name] = (list(array.shape), packed)
            params[index] = torch.from_numpy(packed.view(np.int8).copy())
            weight.tensor_meta["shape"] = [packed.size]
            weight.tensor_meta["dtype"] = TensorDType.Int8
        replacement = MXFP8MatmulOp()
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
        if any(not isinstance(graph.node_table[child], MXFP8MatmulOp) for child in transpose._children):
            raise ValueError(f"MXFP8 weight transpose has another consumer: {transpose_name}")
        graph.node_table[transpose._parents[0]]._children.remove(transpose_name)
        graph._body.remove(transpose)
        del graph.node_table[transpose_name]
    graph._fake_params = [graph._body.index(node) for node in parameters]
    graph._inputs = [graph._body.index(node) for node in parameter_inputs]
    for group in graph.op_groups:
        graph.op_groups[group] = [graph.node_table[node.name] for node in graph.op_groups[group] if node.name not in removed]
    graph._ops_registry["MXFP8MatmulOp"] = lower_matmul
    if {names[positions[node]] for node in quantized} != selected_weights:
        raise ValueError("Selected weights do not match the graph linear operations")
    tensors, weights, floats = [], [], []
    weight_offset = float_offset = 0
    for node, param, parameter_name in zip(parameters, params, names):
        if node.name in quantized:
            shape, packed = quantized[node.name]
            raw = packed.tobytes()
            tensors.append(QuantTensor(parameter_name, shape, [packed.size], "mxfp8", [1],
                                       weight_offset, len(raw), 0, 0,
                                       {"tile_rows": tile_rows, "tile_k": tile_k}))
            weights.append(raw)
            weight_offset += len(raw)
        else:
            raw = param.detach().numpy().tobytes()
            shape = list(param.shape)
            tensors.append(QuantTensor(parameter_name, shape, shape, "f32", [],float_offset,len(raw),0,0))
            floats.append(raw)
            float_offset += len(raw)
    package = RaxQuantPackage(tensors,b"".join(weights),b"".join(floats),b"",{})
    output.mkdir(parents=True, exist_ok=True)
    write_rax(package,output/f"{name}.rax",packer,name)
    (output/"weights.bin").write_bytes(package.weights)
    (output/"params.f32").write_bytes(package.params_f32)

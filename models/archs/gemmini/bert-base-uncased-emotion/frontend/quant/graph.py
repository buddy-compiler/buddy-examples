import torch

from buddy.compiler.graph.graph import NodeType
from buddy.compiler.graph.operation import (
    BatchMatmulOp,
    PlaceholderOp,
    TransposeMatmulFusedOp,
)
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from buddy.compiler.graph.type import TensorDType
from compiler.frontend.quant import (
    FP32ToInt8QuantOp,
    Int32ToFP32DequantOp,
    quantize as quantize_values,
    quantize_bias,
)


def parameter(graph, values, name, value):
    node = PlaceholderOp()
    node.name = name
    node.tensor_meta = {"shape": list(value.shape), "dtype": TensorDType.Float32}
    graph.add_node(node, NodeType.FakeNode)
    values[name] = value
    return node


def operation(graph, op, name, arguments, shape, dtype):
    node = op()
    node.name = name
    for argument in arguments:
        node.add_argument(argument.name)
        node.add_parent(argument.name)
        argument.add_children(name)
    node.tensor_meta = {"shape": list(shape), "dtype": dtype}
    graph.add_node(node)
    return node


def replace_argument(graph, user, index, node):
    source = graph.node_table[user.args[index]]
    source.children.remove(user.name)
    user.parents[user.parents.index(source.name)] = node.name
    user.args[index] = node.name
    node.add_children(user.name)


def replace_uses(graph, source, node):
    """Every user of `source` but `node` reads `node` instead."""
    for name in [name for name in source.children if name != node.name]:
        user = graph.node_table[name]
        for index, argument in enumerate(user.args):
            if argument == source.name:
                user.args[index] = node.name
            elif isinstance(argument, list):
                user.args[index] = [
                    node.name if name == source.name else name for name in argument
                ]
        user._parents = [
            node.name if name == source.name else name for name in user.parents
        ]
        source.children.remove(name)
        node.add_children(name)


def convert(graph, values, op, index, multiplier):
    """Argument `index` of `op` read as INT8 codes: the CPU converts it online,
    times `multiplier`."""
    source = graph.node_table[op.args[index]]
    shape = source.tensor_meta["shape"]
    multiplier = parameter(graph, values, f"{op.name}_{index}_multiplier", multiplier)
    codes = operation(
        graph,
        FP32ToInt8QuantOp,
        f"{op.name}_{index}_codes",
        [source, multiplier],
        shape,
        TensorDType.Int8,
    )
    replace_argument(graph, op, index, codes)


def restore(graph, values, accumulator, step):
    """FP32 values of the INT32 `accumulator`, times `step`, for its users."""
    shape = accumulator.tensor_meta["shape"]
    step = parameter(graph, values, accumulator.name + "_accumulator_step", step)
    restored = operation(
        graph,
        Int32ToFP32DequantOp,
        accumulator.name + "_restored",
        [accumulator, step],
        shape,
        TensorDType.Float32,
    )
    replace_uses(graph, accumulator, restored)


def quantize(graph, params, parameter_names, calibration):
    """Every matrix product runs on the Gemmini INT8 array on INT8 codes and
    accumulates in INT32; the CPU converts the inputs online and restores
    FP32 values. A Linear reads its weight with one scale per output channel
    and its input with one calibrated scale, dividing each input channel by
    its smoothing factor on the way (the weight takes the factor), and
    accumulates on its INT32 bias; an attention product reads both operands
    with one calibrated scale each."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    products = 0
    for op in list(graph.body):
        if isinstance(op, BatchMatmulOp):
            # The attention products in the order the encoder runs them.
            scales = [calibration[f"matmul.{products}.{side}"] / 127 for side in (0, 1)]
            for side, scale in enumerate(scales):
                rank = len(graph.node_table[op.args[side]].tensor_meta["shape"])
                convert(graph, values, op, side, torch.full([1] * rank, 1 / scale))
            op.tensor_meta["dtype"] = TensorDType.Int32
            rank = len(op.tensor_meta["shape"])
            restore(graph, values, op, torch.full([1] * rank, scales[0] * scales[1]))
            products += 1
            continue
        if not isinstance(op, TransposeMatmulFusedOp):
            continue
        weight = graph.node_table[op.args[1]]
        module = parameter_names[weight.name].rsplit(".", 1)[0]
        smooth = torch.tensor(calibration[module]["smooth"])

        # Weights: the smoothing factor along the input channels, axis 1 of the
        # [N, K] weights, then symmetric in [-127, 127] per output channel.
        value = values[weight.name] * smooth.reshape(1, -1)
        step = value.abs().amax(1, keepdim=True) / 127
        values[weight.name] = quantize_values(value, 1 / step)
        weight.tensor_meta["dtype"] = TensorDType.Int8

        # Input: one scale for the smoothed tensor, from the calibrated
        # maximum; the channels are the last axis.
        scale = calibration[module]["maximum"] / 127
        rank = len(graph.node_table[op.args[0]].tensor_meta["shape"])
        convert(
            graph, values, op, 0, 1 / (scale * smooth.reshape([1] * (rank - 1) + [-1]))
        )
        op.tensor_meta["dtype"] = TensorDType.Int32

        # Bias: INT32 in the accumulator, with its per-output-channel scale,
        # the first argument of the add after the product.
        accumulator = graph.node_table[op.children[0]]
        accumulator.tensor_meta["dtype"] = TensorDType.Int32
        bias = graph.node_table[accumulator.args[0]]
        bias_shape = bias.tensor_meta["shape"]
        values[bias.name] = quantize_bias(
            values[bias.name].reshape(-1), (step * scale).reshape(-1)
        ).reshape(bias_shape)
        bias.tensor_meta = {"shape": bias_shape, "dtype": TensorDType.Int32}
        rank = len(accumulator.tensor_meta["shape"])
        restore(
            graph, values, accumulator, (step * scale).reshape([1] * (rank - 1) + [-1])
        )
    sort_graph(graph)
    return [values[node.name] for node in graph.params]

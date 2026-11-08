import torch

from buddy.compiler.graph.graph import NodeType
from buddy.compiler.graph.operation import (
    BatchMatmulOp,
    Conv2dOp,
    PlaceholderOp,
)
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from buddy.compiler.graph.type import TensorDType
from compiler.frontend.quant import (
    FP32ToInt8QuantOp,
    Int8ToFP32DequantOp,
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


def convert(graph, values, op, index, maximum):
    """Argument `index` of `op` read as INT8 values with one scale, from the
    calibrated maximum: it converts to INT8 online."""
    source = graph.node_table[op.args[index]]
    shape = source.tensor_meta["shape"]
    scale = torch.full([1] * len(shape), maximum / 127)
    name = f"{op.name}_{index}"
    multiplier = parameter(graph, values, name + "_multiplier", 1 / scale)
    step = parameter(graph, values, name + "_step", scale)
    codes = operation(
        graph,
        FP32ToInt8QuantOp,
        name + "_codes",
        [source, multiplier],
        shape,
        TensorDType.Int8,
    )
    restored = operation(
        graph,
        Int8ToFP32DequantOp,
        name + "_values",
        [codes, step],
        shape,
        TensorDType.Float32,
    )
    replace_argument(graph, op, index, restored)


def quantize(graph, params, parameter_names, calibration):
    """Every convolution and attention product reads INT8 values. A
    convolution reads its weight with one scale per output channel and its
    input with one calibrated scale; the input converts to INT8 online,
    dividing each input channel by its smoothing factor on the way (the
    weight takes the factor), and its bias is INT32 in the accumulator
    domain. An attention product reads both operands with one calibrated
    scale each."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    products = 0
    for op in list(graph.body):
        if isinstance(op, BatchMatmulOp):
            # The attention products in the order the model runs them.
            convert(graph, values, op, 0, calibration[f"matmul.{products}.0"])
            convert(graph, values, op, 1, calibration[f"matmul.{products}.1"])
            products += 1
            continue
        if not isinstance(op, Conv2dOp):
            continue
        weight = graph.node_table[op.args[1]]
        module = parameter_names[weight.name].rsplit(".", 1)[0]
        smooth = torch.tensor(calibration[module]["smooth"])

        # Weights: the smoothing factor along the input channels, axis 1 of the
        # [F, C, H, W] weights, or axis 0 of the [C, 1, H, W] depthwise ones. Then symmetric in
        # [-127, 127] per output channel, axis 0. A padding channel is zero;
        # any positive step encodes it.
        value = values[weight.name]
        if op.args[-1] > 1:
            value = value * smooth.reshape(-1, 1, 1, 1)
        else:
            value = value * smooth.reshape(1, -1, 1, 1)
        axes = list(range(1, value.ndim))
        maximum = value.abs().amax(axes, keepdim=True)
        step = torch.where(maximum == 0, 1.0, maximum / 127)
        values[weight.name] = quantize_values(value, 1 / step)
        weight.tensor_meta["dtype"] = TensorDType.Int8
        weight_step = parameter(graph, values, weight.name + "_step", step)
        weight_values = operation(
            graph,
            Int8ToFP32DequantOp,
            weight.name + "_dequantized",
            [weight, weight_step],
            weight.tensor_meta["shape"],
            TensorDType.Float32,
        )
        replace_argument(graph, op, 1, weight_values)

        # Input: one scale for the smoothed tensor, from the calibrated
        # maximum; the channels are the last axis of the NHWC input.
        source = graph.node_table[op.args[0]]
        shape = source.tensor_meta["shape"]
        scale = calibration[module]["maximum"] / 127
        multiplier = parameter(
            graph,
            values,
            op.name + "_multiplier",
            1 / (scale * smooth.reshape([1] * (len(shape) - 1) + [-1])),
        )
        input_step = parameter(
            graph, values, op.name + "_step", torch.full([1] * len(shape), scale)
        )
        codes = operation(
            graph,
            FP32ToInt8QuantOp,
            op.name + "_input_codes",
            [source, multiplier],
            shape,
            TensorDType.Int8,
        )
        input_values = operation(
            graph,
            Int8ToFP32DequantOp,
            op.name + "_input",
            [codes, input_step],
            shape,
            TensorDType.Float32,
        )
        replace_argument(graph, op, 0, input_values)

        # Bias: INT32 with the accumulator's per-output-channel scale, the
        # third argument. A convolution before a batch normalization has none.
        if op.args[2] is None:
            continue
        bias = graph.node_table[op.args[2]]
        bias_shape = bias.tensor_meta["shape"]
        bias_step = (step * scale).reshape(bias_shape)
        values[bias.name] = quantize_bias(values[bias.name], bias_step)
        bias.tensor_meta = {"shape": bias_shape, "dtype": TensorDType.Int32}
        bias_step = parameter(graph, values, bias.name + "_step", bias_step)
        bias_values = operation(
            graph,
            Int32ToFP32DequantOp,
            bias.name + "_dequantized",
            [bias, bias_step],
            bias_shape,
            TensorDType.Float32,
        )
        replace_argument(graph, op, 2, bias_values)
    sort_graph(graph)
    return [values[node.name] for node in graph.params]

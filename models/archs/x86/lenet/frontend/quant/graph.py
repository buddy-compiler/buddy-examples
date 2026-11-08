import torch

from buddy.compiler.graph.graph import NodeType
from buddy.compiler.graph.operation import (
    Conv2dOp,
    PlaceholderOp,
    TransposeMatmulFusedOp,
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


def quantize(graph, params, parameter_names, calibration):
    """Every convolution and Linear reads INT8 values: its weight with one
    scale per output channel and its input with one calibrated scale. The
    input converts to INT8 online, dividing each input channel by its
    smoothing factor on the way (the weight takes the factor); its bias is
    INT32 in the accumulator domain."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    for op in list(graph.body):
        if not isinstance(op, (Conv2dOp, TransposeMatmulFusedOp)):
            continue
        weight = graph.node_table[op.args[1]]
        module = parameter_names[weight.name].rsplit(".", 1)[0]
        smooth = torch.tensor(calibration[module]["smooth"])

        # Weights: the smoothing factor along the input channels, axis 1 of the
        # [F, C, H, W] convolution weights and of the [N, K] Linear ones, or
        # axis 0 of the [C, 1, H, W] depthwise ones. Then symmetric in
        # [-127, 127] per output channel, axis 0. A padding channel is zero;
        # any positive step encodes it.
        value = values[weight.name]
        if isinstance(op, Conv2dOp) and op.args[-1] > 1:
            value = value * smooth.reshape(-1, 1, 1, 1)
        else:
            value = value * smooth.reshape([1, -1] + [1] * (value.ndim - 2))
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
        # maximum; the channels are the last axis of the NHWC convolution
        # inputs and of the Linear ones.
        source = graph.node_table[op.args[0]]
        shape = source.tensor_meta["shape"]
        maximum = calibration[module]["maximum"]
        # All-zero inputs (an SE excitation): any positive step encodes them.
        scale = maximum / 127 if maximum else 1.0
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
        # third argument of a convolution, the first of the add after a
        # Linear. A convolution before a batch normalization has none.
        if isinstance(op, TransposeMatmulFusedOp):
            bias_user = graph.node_table[op.children[0]]
            index = 0
        else:
            bias_user = op
            index = 2
        if bias_user.args[index] is None:
            continue
        bias = graph.node_table[bias_user.args[index]]
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
        replace_argument(graph, bias_user, index, bias_values)
    sort_graph(graph)
    return [values[node.name] for node in graph.params]

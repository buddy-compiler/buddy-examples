import torch

from buddy.compiler.graph.graph import NodeType
from buddy.compiler.graph.operation import (
    Conv2dOp,
    MulOp,
    PlaceholderOp,
    TransposeMatmulFusedOp,
    ViewOp,
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


def smooth_classifier(model, smooth):
    """Smooths the classifier Linears through their producers: a Linear
    divides its whole input with one multiplier, so the producer divides its
    output channel by s (its weight and bias; ReLU, pooling and dropout
    between keep the factor) and the Linear weight takes s. classifier.1
    reads the 256 channels of features.10 at 6x6 positions each."""
    pairs = [
        (model.features[10], model.classifier[1], 36, "classifier.1"),
        (model.classifier[1], model.classifier[4], 1, "classifier.4"),
        (model.classifier[4], model.classifier[6], 1, "classifier.6"),
    ]
    with torch.no_grad():
        for producer, linear, positions, name in pairs:
            factor = torch.tensor(smooth[name])
            shape = [-1] + [1] * (producer.weight.ndim - 1)
            producer.weight /= factor.reshape(shape)
            producer.bias /= factor
            linear.weight *= factor.repeat_interleave(positions)


def quantize(graph, params, parameter_names, calibration):
    """Every convolution and Linear reads INT8 values: the weight with one
    scale per output channel, the input with one calibrated scale. The
    weights keep the checkpoint layout: the output channels are axis 0 of the
    [F, C, H, W] convolution weights and of the [N, K] Linear ones, which the
    Linears read transposed. A convolution input converts to INT8 online,
    divided by the smoothing factor of its channel on the way; the weight
    takes the factor. A Linear input converts with one multiplier."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    for op in list(graph.body):
        if not isinstance(op, (Conv2dOp, TransposeMatmulFusedOp)):
            continue
        weight = graph.node_table[op.args[1]]
        module = parameter_names[weight.name].rsplit(".", 1)[0]
        source = graph.node_table[op.args[0]]
        shape = source.tensor_meta["shape"]
        value = values[weight.name]
        if isinstance(op, TransposeMatmulFusedOp):
            # The add after the Linear reads the bias as argument 0.
            bias_user, bias_index = graph.node_table[op.children[0]], 0
            bias = graph.node_table[bias_user.args[bias_index]]

            # Input: one scale for the tensor, from the calibrated maximum.
            scale = torch.full([1] * len(shape), calibration[module] / 127)
            multiplier = 1 / scale

            # Weights: symmetric in [-127, 127] per output channel.
            step = value.abs().amax(1, keepdim=True) / 127
        else:
            # The convolution reads the bias as argument 2.
            bias_user, bias_index = op, 2
            bias = graph.node_table[bias_user.args[bias_index]]

            # Input: one scale for the smoothed tensor, from the calibrated
            # clipping magnitude.
            smooth = torch.tensor(calibration[module]["smooth"])
            scale = torch.full([1] * len(shape), calibration[module]["clip"] / 127)
            # The channels are the last axis of the NHWC input.
            multiplier = 1 / (scale * smooth.reshape([1] * (len(shape) - 1) + [-1]))

            # Weights: the smoothing factor along the input channels, axis 1.
            # Then symmetric in [-127, 127] per output channel. A channel
            # whose folded weights almost vanish takes a coarser step, so
            # that its bias code stays within 2^30.
            value = value * smooth.reshape(1, -1, 1, 1)
            maximum = value.abs().amax([1, 2, 3], keepdim=True)
            floor = values[bias.name].abs().reshape(maximum.shape) / (
                scale.reshape(-1) * 2**30
            )
            step = torch.maximum(maximum / 127, floor)

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

        multiplier = parameter(graph, values, op.name + "_multiplier", multiplier)
        input_step = parameter(graph, values, op.name + "_step", scale)
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

        # Bias shares the INT32 accumulator's per-output-channel scale.
        bias_shape = [value.shape[0]]
        values[bias.name] = quantize_bias(values[bias.name], (step * scale).reshape(-1))
        bias.tensor_meta = {"shape": bias_shape, "dtype": TensorDType.Int32}
        accumulator_step = operation(
            graph,
            MulOp,
            op.name + "_accumulator_step",
            [input_step, weight_step],
            step.shape,
            TensorDType.Float32,
        )
        accumulator_step = operation(
            graph,
            ViewOp,
            op.name + "_bias_step",
            [accumulator_step],
            bias_shape,
            TensorDType.Float32,
        )
        accumulator_step.add_argument(bias_shape)
        bias_values = operation(
            graph,
            Int32ToFP32DequantOp,
            bias.name + "_dequantized",
            [bias, accumulator_step],
            bias_shape,
            TensorDType.Float32,
        )
        replace_argument(graph, bias_user, bias_index, bias_values)
    sort_graph(graph)
    return [values[node.name] for node in graph.params]

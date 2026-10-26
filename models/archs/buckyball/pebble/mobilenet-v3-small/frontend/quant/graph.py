import torch

from buddy.compiler.graph.graph import NodeType
from buddy.compiler.graph.operation import (
    Conv2dOp,
    MatmulOp,
    MulOp,
    PlaceholderOp,
    ViewOp,
)
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from buddy.compiler.graph.type import TensorDType
from stack.compiler.frontend.quant import (
    FP32ToInt8QuantOp,
    Int8ToFP32DequantOp,
    Int32ToFP32DequantOp,
    quantize as quantize_values,
    quantize_bias,
)

# The convolutions that stay in FP32:
# - the depthwise convolutions of the first two blocks read inputs whose
#   channels span very different ranges: one INT8 scale for the tensor loses
#   most of the accuracy;
# - the expansion of block 2 and the depthwise convolution of block 3 would
#   make the pebble core chain a 56x56 or 28x28 image from the convolution
#   before into them, which does not fit its banks;
# - the Linears of the classifier reduce 576 and 1024 inputs, more than one
#   bank holds.
FP32 = [
    "features.1.block.0.0",
    "features.2.block.1.0",
    "features.2.block.0.0",
    "features.3.block.1.0",
    "classifier.0",
    "classifier.3",
]


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


def quantize(graph, params, parameter_names, maxima):
    """Every convolution and Linear but those of FP32 reads INT8 values: its
    weight with one scale per output channel, its input with one calibrated
    scale."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    for op in list(graph.body):
        if not isinstance(op, (Conv2dOp, MatmulOp)):
            continue
        weight = graph.node_table[op.args[1]]
        module = parameter_names[weight.name].rsplit(".", 1)[0]
        if module in FP32:
            continue
        if isinstance(op, Conv2dOp):
            bias_user, bias_index = op, 2
        else:
            bias_user, bias_index = graph.node_table[op.children[0]], 0
        bias = graph.node_table[bias_user.args[bias_index]]

        # Input: one scale for the tensor, from the calibrated clipping
        # magnitude. All-zero inputs: any positive step encodes them.
        source = graph.node_table[op.args[0]]
        shape = source.tensor_meta["shape"]
        clip = maxima[module]
        scale = torch.full([1] * len(shape), clip / 127 if clip else 1.0)

        # Weights: symmetric in [-127, 127] per output channel, axis 0 of the
        # [F, C, H, W] convolution weights and axis 1 of the [K, N] Linear
        # ones. A channel whose folded weights almost vanish takes a coarser
        # step, so that its bias code stays within 2^30. A padding channel is
        # zero; any positive step encodes it.
        value = values[weight.name]
        channel = 0 if isinstance(op, Conv2dOp) else 1
        axes = [axis for axis in range(value.ndim) if axis != channel]
        maximum = value.abs().amax(axes, keepdim=True)
        floor = values[bias.name].abs().reshape(maximum.shape) / (
            scale.reshape(-1) * 2**30
        )
        step = torch.maximum(maximum / 127, floor)
        step = torch.where(step == 0, 1.0, step)
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

        # Input codes.
        multiplier = parameter(graph, values, op.name + "_multiplier", 1 / scale)
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
        bias_shape = (
            [value.shape[0]] if isinstance(op, Conv2dOp) else [1, value.shape[1]]
        )
        values[bias.name] = quantize_bias(
            values[bias.name].reshape(-1), (step * scale).reshape(-1)
        ).reshape(bias_shape)
        bias.tensor_meta = {"shape": bias_shape, "dtype": TensorDType.Int32}
        accumulator_step = operation(
            graph,
            MulOp,
            op.name + "_accumulator_step",
            [input_step, weight_step],
            step.shape,
            TensorDType.Float32,
        )
        if isinstance(op, Conv2dOp):
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

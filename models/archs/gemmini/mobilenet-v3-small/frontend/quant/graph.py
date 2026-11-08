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


def quantize(graph, params, parameter_names, calibration):
    """Every convolution and Linear runs on the Gemmini INT8 array: it reads
    INT8 codes, its weight with one scale per output channel and its input
    with one calibrated scale, and accumulates on its INT32 bias. The CPU
    converts the input to INT8 online, dividing each input channel by its
    smoothing factor on the way (the weight takes the factor), and restores
    FP32 values from the INT32 accumulator."""
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
        codes = operation(
            graph,
            FP32ToInt8QuantOp,
            op.name + "_input_codes",
            [source, multiplier],
            shape,
            TensorDType.Int8,
        )
        replace_argument(graph, op, 0, codes)
        op.tensor_meta["dtype"] = TensorDType.Int32

        # Bias: INT32 in the accumulator, with its per-output-channel scale:
        # the third argument of a convolution, the first of the add after a
        # Linear. A convolution before a batch normalization has none.
        accumulator = op
        bias = None
        if isinstance(op, TransposeMatmulFusedOp):
            accumulator = graph.node_table[op.children[0]]
            accumulator.tensor_meta["dtype"] = TensorDType.Int32
            bias = graph.node_table[accumulator.args[0]]
        elif op.args[2] is not None:
            bias = graph.node_table[op.args[2]]
        if bias is not None:
            bias_shape = bias.tensor_meta["shape"]
            values[bias.name] = quantize_bias(
                values[bias.name].reshape(-1), (step * scale).reshape(-1)
            ).reshape(bias_shape)
            bias.tensor_meta = {"shape": bias_shape, "dtype": TensorDType.Int32}

        # FP32 values: the accumulator times input step times weight step, one
        # per output channel, the last axis of the output.
        output_shape = accumulator.tensor_meta["shape"]
        accumulator_step = parameter(
            graph,
            values,
            op.name + "_accumulator_step",
            (step * scale).reshape([1] * (len(output_shape) - 1) + [-1]),
        )
        restored = operation(
            graph,
            Int32ToFP32DequantOp,
            op.name + "_restored",
            [accumulator, accumulator_step],
            output_shape,
            TensorDType.Float32,
        )
        replace_uses(graph, accumulator, restored)
    sort_graph(graph)
    return [values[node.name] for node in graph.params]

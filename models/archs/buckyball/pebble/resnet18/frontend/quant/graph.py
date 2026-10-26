import torch

from buddy.compiler.graph.graph import NodeType
from buddy.compiler.graph.operation import MatmulOp, MulOp, PlaceholderOp
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from buddy.compiler.graph.type import TensorDType
from stack.compiler.frontend.quant import (
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
    user.args[index] = node.name
    user.parents[user.parents.index(source.name)] = node.name
    node.add_children(user.name)


def quantize(graph, params, parameter_names, maxima):
    """The fc Linear reads INT8 values: its weight with one scale per output
    channel, its input with one calibrated scale. The convolutions stay FP32:
    the pebble core unrolls every 16-channel term of every output tile into
    bank operations, far too many for the ImageNet feature maps."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    for op in list(graph.body):
        if not isinstance(op, MatmulOp):
            continue
        weight = graph.node_table[op.args[1]]
        module = parameter_names[weight.name].rsplit(".", 1)[0]

        # Weights: symmetric in [-127, 127] per output channel, axis 1 of the
        # [K, N] weights.
        value = values[weight.name]
        step = value.abs().amax(0, keepdim=True) / 127
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

        # Input: one scale for the tensor, from the calibrated maximum.
        source = graph.node_table[op.args[0]]
        shape = source.tensor_meta["shape"]
        scale = torch.full([1] * len(shape), maxima[module] / 127)
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
        bias_user = graph.node_table[op.children[0]]
        bias = graph.node_table[bias_user.args[0]]
        bias_shape = [1, value.shape[1]]
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
        bias_values = operation(
            graph,
            Int32ToFP32DequantOp,
            bias.name + "_dequantized",
            [bias, accumulator_step],
            bias_shape,
            TensorDType.Float32,
        )
        replace_argument(graph, bias_user, 0, bias_values)
    sort_graph(graph)
    return [values[node.name] for node in graph.params]

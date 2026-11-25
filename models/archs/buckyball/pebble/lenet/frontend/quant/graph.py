import torch

from buddy.compiler.graph.graph import NodeType
from buddy.compiler.graph.operation import (
    Conv2dOp,
    MatmulOp,
    MulOp,
    PermuteOp,
    PlaceholderOp,
    SliceOp,
    TensorConstantOp,
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


def parameter(graph, values, name, value, constant):
    node = TensorConstantOp() if constant else PlaceholderOp()
    node.name = name
    node.tensor_meta = {"shape": list(value.shape), "dtype": TensorDType.Float32}
    if constant:
        node.add_argument(
            value.item() if value.numel() == 1 else value.detach().numpy()
        )
        graph.add_node(node)
    else:
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


def conv_layout(graph, values, weight):
    """The INT8 [F, C, H, W] convolution weight stored offline as the B
    operand of the pebble matmul, which the compiler moves in place: S =
    ceil(F / 16) slices of sixteen output channels, C zero-padded to B blocks
    of sixteen and split in chunks of c blocks, c the most, at most eight,
    that divide B (convChunk of the compiler); [S, B / c, H, W, 16 * c, 16]
    stored as [S, rest], line (y, x, i) of a chunk the sixteen output
    channels of input channel i at operator position (y, x). Returns the node
    of the [F, C, H, W] weight the graph reads: the stored parameter viewed
    back, a transpose and zero-offset slices."""
    f, c, h, w = weight.tensor_meta["shape"]
    slices, blocks = (f + 15) // 16, (c + 15) // 16
    chunk = max(d for d in range(1, 9) if blocks % d == 0)
    chunks = blocks // chunk
    value = torch.nn.functional.pad(
        values[weight.name], (0, 0, 0, 0, 0, blocks * 16 - c, 0, slices * 16 - f)
    )
    values[weight.name] = (
        value.reshape(slices, 16, chunks, chunk * 16, h, w)
        .permute(0, 2, 4, 5, 3, 1)
        .contiguous()
        .reshape(slices, -1)
    )
    weight.tensor_meta["shape"] = list(values[weight.name].shape)
    int8 = TensorDType.Int8
    name = weight.name
    shape = [slices, chunks, h, w, chunk * 16, 16]
    node = operation(graph, ViewOp, name + "_chunks", [weight], shape, int8)
    node.add_argument(shape)
    shape = [slices, 16, chunks, chunk * 16, h, w]
    node = operation(graph, PermuteOp, name + "_lines", [node], shape, int8)
    node.add_argument([0, 5, 1, 4, 2, 3])
    shape = [slices * 16, blocks * 16, h, w]
    node = operation(graph, ViewOp, name + "_padded", [node], shape, int8)
    node.add_argument(shape)
    for dim, size, full in ((0, f, slices * 16), (1, c, blocks * 16)):
        if size == full:
            continue
        shape = [f, blocks * 16, h, w] if dim == 0 else [f, c, h, w]
        node = operation(graph, SliceOp, f"{name}_slice{dim}", [node], shape, int8)
        for argument in (dim, 0, size):
            node.add_argument(argument)
    return node


def quantize(graph, params, parameter_names, maxima):
    """Every convolution and Linear reads INT8 values: its weight with one
    scale per output channel, its input with one calibrated scale."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    for op in list(graph.body):
        if not isinstance(op, (Conv2dOp, MatmulOp)):
            continue
        weight = graph.node_table[op.args[1]]
        module = parameter_names[weight.name].rsplit(".", 1)[0]

        # Weights: symmetric in [-127, 127] per output channel, axis 0 of the
        # [F, C, H, W] convolution weights and axis 1 of the [K, N] Linear
        # ones. A padding channel is zero; any positive step encodes it.
        value = values[weight.name]
        channel = 0 if isinstance(op, Conv2dOp) else 1
        axes = [axis for axis in range(value.ndim) if axis != channel]
        maximum = value.abs().amax(axes, keepdim=True)
        step = torch.where(maximum == 0, 1.0, maximum / 127)
        values[weight.name] = quantize_values(value, 1 / step)
        weight.tensor_meta["dtype"] = TensorDType.Int8
        weight_step = parameter(
            graph, values, weight.name + "_step", step, isinstance(op, MatmulOp)
        )
        weight_shape = weight.tensor_meta["shape"]
        weight_codes = weight
        if isinstance(op, Conv2dOp) and op.args[-1] == 1:
            weight_codes = conv_layout(graph, values, weight)
        elif isinstance(op, MatmulOp) and weight_shape[1] > 16:
            k, n = weight_shape
            values[weight.name] = (
                values[weight.name]
                .reshape(k, n // 16, 16)
                .permute(1, 0, 2)
                .contiguous()
            )
            weight.tensor_meta["shape"] = [n // 16, k, 16]
            columns = operation(
                graph,
                PermuteOp,
                weight.name + "_columns",
                [weight],
                [k, n // 16, 16],
                TensorDType.Int8,
            )
            columns.add_argument([1, 0, 2])
            weight_codes = operation(
                graph,
                ViewOp,
                weight.name + "_matrix",
                [columns],
                weight_shape,
                TensorDType.Int8,
            )
            weight_codes.add_argument(weight_shape)
        weight_values = operation(
            graph,
            Int8ToFP32DequantOp,
            weight.name + "_dequantized",
            [weight_codes, weight_step],
            weight_shape,
            TensorDType.Float32,
        )
        replace_argument(graph, op, 1, weight_values)

        # Input: one scale for the tensor, from the calibrated maximum.
        source = graph.node_table[op.args[0]]
        shape = source.tensor_meta["shape"]
        scale = torch.full([1] * len(shape), maxima[module] / 127)
        multiplier = parameter(
            graph, values, op.name + "_multiplier", 1 / scale, isinstance(op, MatmulOp)
        )
        input_step = parameter(
            graph, values, op.name + "_step", scale, isinstance(op, MatmulOp)
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

        # Bias shares the INT32 accumulator's per-output-channel scale.
        if isinstance(op, Conv2dOp):
            bias_user, bias_index = op, 2
            bias_shape = [value.shape[0]]
        else:
            bias_user, bias_index = graph.node_table[op.children[0]], 0
            bias_shape = [1, value.shape[1]]
        bias = graph.node_table[bias_user.args[bias_index]]
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

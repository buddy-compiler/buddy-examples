import torch

from buddy.compiler.graph.graph import NodeType
from buddy.compiler.graph.operation import (
    Conv2dOp,
    MatmulOp,
    MulOp,
    PermuteOp,
    PlaceholderOp,
    SliceOp,
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
    node.tensor_meta = {
        "shape": list(value.shape),
        "dtype": {
            torch.float32: TensorDType.Float32,
            torch.int32: TensorDType.Int32,
            torch.int8: TensorDType.Int8,
        }[value.dtype],
    }
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


def depthwise_layout(graph, values, weight):
    channels, _, height, width = weight.tensor_meta["shape"]
    blocks = (channels + 15) // 16
    diagonal = torch.zeros((blocks, height, width, 16, 16), dtype=torch.int8)
    for channel in range(channels):
        diagonal[channel // 16, :, :, channel % 16, channel % 16] = values[weight.name][
            channel, 0
        ]
    values[weight.name] = diagonal.reshape(blocks, -1)
    weight.tensor_meta["shape"] = list(values[weight.name].shape)
    name, dtype = weight.name, TensorDType.Int8
    shape = [blocks, height, width, 256]
    node = operation(graph, ViewOp, name + "_diagonal_blocks", [weight], shape, dtype)
    node.add_argument(shape)
    node = operation(
        graph, SliceOp, name + "_diagonal", [node], [blocks, height, width, 16], dtype
    )
    for argument in (3, 0, 256, 17):
        node.add_argument(argument)
    node = operation(
        graph, PermuteOp, name + "_channels", [node], [blocks, 16, height, width], dtype
    )
    node.add_argument([0, 3, 1, 2])
    shape = [blocks * 16, 1, height, width]
    node = operation(graph, ViewOp, name + "_padded", [node], shape, dtype)
    node.add_argument(shape)
    if channels != blocks * 16:
        node = operation(
            graph,
            SliceOp,
            name + "_trimmed",
            [node],
            [channels, 1, height, width],
            dtype,
        )
        for argument in (0, 0, channels):
            node.add_argument(argument)
    return node


def quantize(graph, params, parameter_names, calibration):
    """Every convolution and Linear reads INT8 values: its weight with one scale per
    output channel, its input with one calibrated scale. The input converts
    to INT8 online, divided by the smoothing factor of its channel on the way;
    the weight takes the factor."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    for op in list(graph.body):
        if not isinstance(op, (Conv2dOp, MatmulOp)):
            continue
        weight = graph.node_table[op.args[1]]
        module = parameter_names[weight.name].rsplit(".", 1)[0]
        smooth = torch.tensor(calibration[module]["smooth"])
        convolution = isinstance(op, Conv2dOp)
        bias_user, bias_index = (
            (op, 2) if convolution else (graph.node_table[op.children[0]], 0)
        )
        bias = graph.node_table[bias_user.args[bias_index]]

        # Input: one scale for the smoothed tensor, from the calibrated
        # clipping magnitude. All-zero inputs (an SE excitation): any positive
        # step encodes them.
        source = graph.node_table[op.args[0]]
        shape = source.tensor_meta["shape"]
        clip = calibration[module]["clip"]
        scale = torch.full([1] * len(shape), clip / 127 if clip else 1.0)

        # Weights: the smoothing factor along the input channels, axis 1 of
        # the [F, C, H, W] weights or axis 0 of the [C, 1, H, W] depthwise
        # ones. Then symmetric in [-127, 127] per output channel, axis 0. A
        # channel whose folded weights almost vanish takes a coarser step, so
        # that its bias code stays within 2^30.
        value = values[weight.name]
        if not convolution:
            value = value * smooth.reshape(-1, 1)
        elif op.args[-1] > 1:
            value = value * smooth.reshape(-1, 1, 1, 1)
        else:
            value = value * smooth.reshape(1, -1, 1, 1)
        maximum = value.abs().amax([1, 2, 3] if convolution else [0], keepdim=True)
        floor = values[bias.name].abs().reshape(maximum.shape) / (
            scale.reshape(-1) * 2**30
        )
        step = torch.maximum(maximum / 127, floor)
        step = torch.where(step == 0, 1.0, step)
        values[weight.name] = quantize_values(value, 1 / step)
        weight.tensor_meta["dtype"] = TensorDType.Int8
        weight_step = parameter(graph, values, weight.name + "_step", step)
        weight_shape = weight.tensor_meta["shape"]
        weight_codes = weight
        if isinstance(op, Conv2dOp):
            weight_codes = (
                conv_layout(graph, values, weight)
                if op.args[-1] == 1
                else depthwise_layout(graph, values, weight)
            )
        else:
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

        # Input codes: the channels are the last axis of the NHWC input.
        multiplier = parameter(
            graph,
            values,
            op.name + "_multiplier",
            1 / (scale * smooth.reshape([1] * (len(shape) - 1) + [-1])),
        )
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
        bias_shape = [value.shape[0]] if convolution else [1, value.shape[1]]
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
        if convolution:
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

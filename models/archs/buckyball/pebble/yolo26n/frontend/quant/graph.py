import torch
from models.archs.buckyball.pebble.fixed import pad_quantization

from buddy.compiler.graph.graph import NodeType
from buddy.compiler.graph.operation import (
    AddOp,
    AliasOp,
    BatchMatmulOp,
    CatOp,
    Conv2dOp,
    ExpandOp,
    GetItemOp,
    MulOp,
    MaxPool2dOp,
    OutputOp,
    PermuteOp,
    PlaceholderOp,
    SliceOp,
    SiluOp,
    SplitOp,
    TransposeOp,
    UnsafeIndexOp,
    ViewOp,
)
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from buddy.compiler.graph.type import TensorDType
from compiler.frontend.quant import (
    FP32ToInt8QuantOp,
    Int8ToFP32DequantOp,
    Int32ToFP32DequantOp,
    Int32ToInt8QuantOp,
    LookupI8Op,
    ScaledAddI8Op,
    quantize as quantize_values,
    quantize_bias,
)


def parameter(graph, values, name, value):
    node = PlaceholderOp()
    node.name = name
    node.tensor_meta = {
        "shape": list(value.shape),
        "dtype": {torch.float32: TensorDType.Float32, torch.int8: TensorDType.Int8}[
            value.dtype
        ],
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


def replace_result(graph, node, replacement):
    users = list(node.children)
    graph.replace_as_parent(node, users, replacement)
    for name in users:
        user = graph.node_table[name]
        if isinstance(user, CatOp):
            user.args[0] = [
                replacement.name if argument == node.name else argument
                for argument in user.args[0]
            ]
        replacement.add_children(name)
    graph.delete_node(node, [graph.node_table[parent] for parent in node.parents])


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
    """Every convolution reads INT8 values: its weight with one scale per
    output channel, its input with one calibrated scale. The input converts
    to INT8 online, divided by the smoothing factor of its channel on the way;
    the weight takes the factor. An attention product reads both operands
    with one calibrated scale each, numbered in the order the model runs
    them."""
    # Cancel layout round trips before composing integer lookup tables.
    for node in list(graph.body):
        if not isinstance(node, PermuteOp):
            continue
        source = graph.node_table[node.args[0]]
        while isinstance(source, AliasOp):
            source = graph.node_table[source.args[0]]
        if not isinstance(source, PermuteOp):
            continue
        permutation = [source.args[1][axis] for axis in node.args[1]]
        if permutation == list(range(len(permutation))):
            replace_result(graph, node, graph.node_table[source.args[0]])
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    products = 0
    activation_steps = {}
    for op in list(graph.body):
        if isinstance(op, BatchMatmulOp):
            users = list(op.children)
            steps = []
            for side in (0, 1):
                source = graph.node_table[op.args[side]]
                layouts = []
                while isinstance(source, (ViewOp, ExpandOp, PermuteOp, TransposeOp)):
                    layouts.append(source)
                    source = graph.node_table[source.args[0]]
                shape = source.tensor_meta["shape"]
                scale = torch.full(
                    [1] * len(shape), calibration[f"matmul.{products}.{side}"] / 127
                )
                name = f"{op.name}_{side}"
                multiplier = parameter(graph, values, name + "_multiplier", 1 / scale)
                steps.append(scale)
                codes = operation(
                    graph,
                    FP32ToInt8QuantOp,
                    name + "_codes",
                    [source, multiplier],
                    shape,
                    TensorDType.Int8,
                )
                for layout in reversed(layouts):
                    if isinstance(layout, ExpandOp):
                        if tuple(codes.tensor_meta["shape"]) != tuple(
                            layout.tensor_meta["shape"]
                        ):
                            raise ValueError(
                                "Attention expansion must preserve the tensor shape"
                            )
                        continue
                    codes = operation(
                        graph,
                        type(layout),
                        name + "_" + layout.name,
                        [codes],
                        layout.tensor_meta["shape"],
                        TensorDType.Int8,
                    )
                    for argument in layout.args[1:]:
                        codes.add_argument(argument)
                    codes.kwargs.update(layout.kwargs)
                replace_argument(graph, op, side, codes)
            op.tensor_meta["dtype"] = TensorDType.Int32
            step = parameter(
                graph,
                values,
                op.name + "_step",
                (steps[0].reshape(()) * steps[1].reshape(())).reshape(1, 1, 1),
            )
            restored = operation(
                graph,
                Int32ToFP32DequantOp,
                op.name + "_values",
                [op, step],
                op.tensor_meta["shape"],
                TensorDType.Float32,
            )
            graph.replace_as_parent(op, users, restored)
            restored.children[:] = users
            op.children[:] = [restored.name]
            products += 1
            continue
        if not isinstance(op, Conv2dOp):
            continue
        weight = graph.node_table[op.args[1]]
        module = parameter_names[weight.name].rsplit(".", 1)[0]
        activation_steps[op.name] = (
            calibration[module]["silu_input_step"],
            calibration[module]["silu_output_step"],
        )
        smooth = torch.tensor(calibration[module]["smooth"])
        bias = graph.node_table[op.args[2]]

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
        if op.args[-1] > 1:
            value = value * smooth.reshape(-1, 1, 1, 1)
        else:
            value = value * smooth.reshape(1, -1, 1, 1)
        maximum = value.abs().amax([1, 2, 3], keepdim=True)
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
        weight_codes = (
            conv_layout(graph, values, weight)
            if op.args[-1] == 1
            else depthwise_layout(graph, values, weight)
        )
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
        bias_shape = [value.shape[0]]
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
        replace_argument(graph, op, 2, bias_values)
    residual = 0
    for node in list(graph.body):
        if not isinstance(node, AddOp) or len(node.tensor_meta["shape"]) != 4:
            continue
        if len(node.parents) != 2:
            continue
        sources = [graph.node_table[name] for name in node.args[:2]]
        shape = list(node.tensor_meta["shape"])
        if any(list(source.tensor_meta["shape"]) != shape for source in sources):
            continue
        number = residual
        residual += 1
        if number in (7, 17):
            continue
        assert node._layout == "NHWC"
        steps = [
            torch.tensor(calibration["residual_steps"][f"{number}.{side}"])
            for side in range(3)
        ]
        matrix_shape = [shape[0] * shape[1] * shape[2], shape[3]]
        inputs = []
        for side, source in enumerate(sources):
            multiplier = parameter(
                graph,
                values,
                f"{node.name}_multiplier{side}",
                (1 / steps[side]).reshape(1, 1, 1, 1),
            )
            codes = operation(
                graph,
                FP32ToInt8QuantOp,
                f"{node.name}_codes{side}",
                [source, multiplier],
                shape,
                TensorDType.Int8,
            )
            matrix = operation(
                graph,
                ViewOp,
                f"{node.name}_matrix{side}",
                [codes],
                matrix_shape,
                TensorDType.Int8,
            )
            matrix.add_argument(matrix_shape)
            inputs.append(matrix)
        added = operation(
            graph,
            ScaledAddI8Op,
            node.name + "_i8",
            inputs,
            matrix_shape,
            TensorDType.Int8,
        )
        added.add_argument((steps[0] / steps[2]).item())
        added.add_argument((steps[1] / steps[2]).item())
        codes = operation(
            graph, ViewOp, node.name + "_nhwc", [added], shape, TensorDType.Int8
        )
        codes.add_argument(shape)
        step = parameter(
            graph, values, node.name + "_step", steps[2].reshape(1, 1, 1, 1)
        )
        decoded = operation(
            graph,
            Int8ToFP32DequantOp,
            node.name + "_values",
            [codes, step],
            shape,
            TensorDType.Float32,
        )
        replace_result(graph, node, decoded)
    assert residual == 20, residual

    for node in list(graph.body):
        if not isinstance(node, SiluOp):
            continue
        source = graph.node_table[node.args[0]]
        shape = list(node.tensor_meta["shape"])
        input_step = torch.tensor(activation_steps[source.name][0], dtype=torch.float32)
        output_step = torch.tensor(
            activation_steps[source.name][1], dtype=torch.float32
        )
        channel_shape = [1, 1, 1, shape[3]]
        scale = parameter(
            graph, values, node.name + "_step", output_step.reshape(channel_shape)
        )
        multiplier = parameter(
            graph,
            values,
            node.name + "_multiplier",
            (1 / input_step).reshape(channel_shape),
        )
        codes = operation(
            graph,
            FP32ToInt8QuantOp,
            node.name + "_codes",
            [source, multiplier],
            shape,
            TensorDType.Int8,
        )
        matrix_shape = [shape[0] * shape[1] * shape[2], shape[3]]
        matrix = operation(
            graph,
            ViewOp,
            node.name + "_matrix",
            [codes],
            matrix_shape,
            TensorDType.Int8,
        )
        matrix.add_argument(matrix_shape)
        indices = torch.arange(256)
        inputs = (
            torch.where(indices < 128, indices, indices - 256).float()[None, :]
            * input_step[:, None]
        )
        table = parameter(
            graph,
            values,
            node.name + "_table",
            torch.nn.functional.pad(
                quantize_values(
                    torch.nn.functional.silu(inputs), 1 / output_step[:, None]
                ),
                (0, 0, 0, -shape[3] % 16),
            ).flatten(),
        )
        looked = operation(
            graph,
            LookupI8Op,
            node.name + "_lookup",
            [matrix, table],
            matrix_shape,
            TensorDType.Int8,
        )
        restored = operation(
            graph, ViewOp, node.name + "_nhwc", [looked], shape, TensorDType.Int8
        )
        restored.add_argument(shape)
        decoded = operation(
            graph,
            Int8ToFP32DequantOp,
            node.name + "_values",
            [restored, scale],
            shape,
            TensorDType.Float32,
        )
        remaining = []
        for name in list(node.children):
            user = graph.node_table[name]
            if not isinstance(user, FP32ToInt8QuantOp):
                remaining.append(name)
                continue
            target_multiplier = graph.node_table[user.args[1]]
            restored_values = (
                values[table.name].reshape(-1, 256)[: shape[3]].float()
                * output_step[:, None]
            )
            target_table = parameter(
                graph,
                values,
                user.name + "_table",
                torch.nn.functional.pad(
                    quantize_values(
                        restored_values, values[target_multiplier.name].reshape(-1, 1)
                    ),
                    (0, 0, 0, -shape[3] % 16),
                ).flatten(),
            )
            target = operation(
                graph,
                LookupI8Op,
                user.name + "_lookup",
                [matrix, target_table],
                matrix_shape,
                TensorDType.Int8,
            )
            target_view = operation(
                graph, ViewOp, user.name + "_nhwc", [target], shape, TensorDType.Int8
            )
            target_view.add_argument(shape)
            users = list(user.children)
            graph.replace_as_parent(user, users, target_view)
            for child in users:
                target_view.add_children(child)
            graph.delete_node(user, [node, target_multiplier])
        graph.replace_as_parent(node, remaining, decoded)
        for user in remaining:
            consumer = graph.node_table[user]
            if isinstance(consumer, CatOp):
                consumer.args[0] = [
                    decoded.name if name == node.name else name
                    for name in consumer.args[0]
                ]
            decoded.add_children(user)
        graph.delete_node(node, [source])

    # Slice codes before dequantizing, so split views do not copy FP32 maps.
    for node in list(graph.body):
        if not isinstance(node, GetItemOp):
            continue
        split = graph.node_table[node.args[0]]
        if not isinstance(split, SplitOp):
            continue
        decoded = graph.node_table[split.args[0]]
        if not isinstance(decoded, Int8ToFP32DequantOp):
            continue
        shape = node.tensor_meta["shape"]
        axis = split.args[2] % len(shape)
        widths = split.tensor_meta["shape"]
        start = sum(part[axis] for part in widths[: node.args[1]])
        codes = operation(
            graph,
            SliceOp,
            node.name + "_codes",
            [graph.node_table[decoded.args[0]]],
            shape,
            TensorDType.Int8,
        )
        for argument in (axis, start, start + shape[axis]):
            codes.add_argument(argument)
        step = values[decoded.args[1]]
        if step.shape[axis] != 1:
            step = step.narrow(axis, start, shape[axis]).contiguous()
        parameter_step = parameter(graph, values, node.name + "_step", step)
        replacement = operation(
            graph,
            Int8ToFP32DequantOp,
            node.name + "_values",
            [codes, parameter_step],
            shape,
            TensorDType.Float32,
        )
        replace_result(graph, node, replacement)

    # Quantization commutes with concatenation; each branch keeps its exact
    # channel multipliers and only the byte tensors need concatenating.
    for node in list(graph.body):
        if not isinstance(node, FP32ToInt8QuantOp):
            continue
        cat = graph.node_table[node.args[0]]
        if not isinstance(cat, CatOp):
            continue
        shape = cat.tensor_meta["shape"]
        axis = cat.args[1] % len(shape)
        factors = values[node.args[1]]
        inputs = []
        start = 0
        for number, name in enumerate(cat.args[0]):
            source = graph.node_table[name]
            part_shape = source.tensor_meta["shape"]
            factor = factors
            if factors.shape[axis] != 1:
                factor = factors.narrow(axis, start, part_shape[axis]).contiguous()
            multiplier = parameter(
                graph, values, f"{node.name}_part{number}_multiplier", factor
            )
            codes = operation(
                graph,
                FP32ToInt8QuantOp,
                f"{node.name}_part{number}",
                [source, multiplier],
                part_shape,
                TensorDType.Int8,
            )
            inputs.append(codes)
            start += part_shape[axis]
        replacement = CatOp()
        replacement.name = node.name + "_cat"
        replacement.tensor_meta = {"shape": list(shape), "dtype": TensorDType.Int8}
        replacement.add_argument([part.name for part in inputs])
        replacement.add_argument(axis)
        for part in inputs:
            replacement.add_parent(part.name)
            part.add_children(replacement.name)
        graph.add_node(replacement)
        replace_result(graph, node, replacement)

    # Positive channel quantization commutes with spatial selection and max.
    # Keep those operations byte-sized before composing the input lookup.
    pending = [node for node in graph.body if isinstance(node, FP32ToInt8QuantOp)]
    for node in pending:
        source = graph.node_table[node.args[0]]
        if not isinstance(source, (UnsafeIndexOp, MaxPool2dOp)):
            continue
        if isinstance(source, UnsafeIndexOp):
            indices = source.args[1]
            assert len(indices) == 4 and indices[0] is None and indices[3] is None
        factors = values[node.args[1]]
        assert torch.all(torch.isfinite(factors) & (factors > 0))
        assert all(size == 1 for size in factors.shape[:-1])
        original = graph.node_table[source.args[0]]
        codes = operation(
            graph,
            FP32ToInt8QuantOp,
            node.name + "_before_spatial",
            [original, graph.node_table[node.args[1]]],
            original.tensor_meta["shape"],
            TensorDType.Int8,
        )
        pending.append(codes)
        if isinstance(source, MaxPool2dOp):
            replacement = operation(
                graph,
                MaxPool2dOp,
                node.name + "_pool",
                [codes],
                node.tensor_meta["shape"],
                TensorDType.Int8,
            )
            replacement._layout = source._layout
            for arg in source.args[1:]:
                replacement.add_argument(arg)
        else:
            replacement = UnsafeIndexOp()
            replacement.name = node.name + "_index"
            replacement.tensor_meta = {
                "shape": list(node.tensor_meta["shape"]),
                "dtype": TensorDType.Int8,
            }
            replacement.add_argument(codes.name)
            replacement.add_argument(list(source.args[1]))
            for parent in [codes.name, *source.parents[1:]]:
                replacement.add_parent(parent)
                graph.node_table[parent].add_children(replacement.name)
            graph.add_node(replacement)
        replacement.kwargs.update(source.kwargs)
        replace_result(graph, node, replacement)

    # Compose a dequantize/requantize pair into its producer's lookup table.
    # The two FP32 products are evaluated in their original order offline.
    for node in list(graph.body):
        if not isinstance(node, FP32ToInt8QuantOp):
            continue
        decoded = graph.node_table[node.args[0]]
        if not isinstance(decoded, Int8ToFP32DequantOp):
            continue
        codes = graph.node_table[decoded.args[0]]
        shape = node.tensor_meta["shape"]
        first, channels = 0, shape[-1]
        if isinstance(codes, SliceOp):
            if codes.args[1] % len(shape) != len(shape) - 1:
                continue
            first = codes.args[2]
            codes = graph.node_table[codes.args[0]]
        while isinstance(codes, ViewOp):
            codes = graph.node_table[codes.args[0]]
        if not isinstance(codes, LookupI8Op):
            original = graph.node_table[decoded.args[0]]
            matrix_shape = [shape[0] * shape[1] * shape[2], channels]
            matrix = operation(
                graph,
                ViewOp,
                node.name + "_matrix",
                [original],
                matrix_shape,
                TensorDType.Int8,
            )
            matrix.add_argument(matrix_shape)
            indices = torch.arange(256)
            signed = torch.where(indices < 128, indices, indices - 256).float()[None, :]
            step = values[decoded.args[1]].reshape(-1, 1)
            multiplier = values[node.args[1]].reshape(-1, 1)
            table = quantize_values(signed * step, multiplier).expand(channels, 256)
            table = torch.nn.functional.pad(table, (0, 0, 0, -channels % 16)).flatten()
            table_node = parameter(graph, values, node.name + "_table", table)
            lookup = operation(
                graph,
                LookupI8Op,
                node.name + "_lookup",
                [matrix, table_node],
                matrix_shape,
                TensorDType.Int8,
            )
            replacement = operation(
                graph, ViewOp, node.name + "_view", [lookup], shape, TensorDType.Int8
            )
            replacement.add_argument(shape)
            replace_result(graph, node, replacement)
            continue
        matrix = graph.node_table[codes.args[0]]
        rows, full_channels = matrix.tensor_meta["shape"]
        old_table = values[codes.args[1]].reshape(-1, 256)
        if old_table.shape[0] == 1:
            old_table = old_table.expand(full_channels, 256)
        old_table = old_table[first : first + channels].float()
        step = values[decoded.args[1]].reshape(-1, 1)
        multiplier = values[node.args[1]].reshape(-1, 1)
        table = quantize_values(old_table * step, multiplier)
        table = torch.nn.functional.pad(table, (0, 0, 0, -channels % 16)).flatten()
        table_node = parameter(graph, values, node.name + "_table", table)
        if first != 0 or channels != full_channels:
            matrix = operation(
                graph,
                SliceOp,
                node.name + "_input_slice",
                [matrix],
                [rows, channels],
                TensorDType.Int8,
            )
            for argument in (1, first, first + channels):
                matrix.add_argument(argument)
        lookup = operation(
            graph,
            LookupI8Op,
            node.name + "_lookup",
            [matrix, table_node],
            [rows, channels],
            TensorDType.Int8,
        )
        replacement = operation(
            graph, ViewOp, node.name + "_view", [lookup], shape, TensorDType.Int8
        )
        replacement.add_argument(shape)
        replace_result(graph, node, replacement)

    # Keep integer accumulation explicit in both the reference and NPU graph.
    for node in list(graph.body):
        if not isinstance(node, Conv2dOp):
            continue
        users = list(node.children)
        for index in (0, 1, 2):
            decoded = graph.node_table[node.args[index]]
            if index == 2:
                scale = graph.node_table[decoded.args[1]]
                assert isinstance(scale, ViewOp)
                scale = graph.node_table[scale.args[0]]
                assert isinstance(scale, MulOp)
                step = (values[scale.args[0]] * values[scale.args[1]]).reshape(
                    1, 1, 1, -1
                )
            replace_argument(graph, node, index, graph.node_table[decoded.args[0]])
        node.tensor_meta["dtype"] = TensorDType.Int32
        scale = parameter(graph, values, node.name + "_output_step", step)
        decoded = operation(
            graph,
            Int32ToFP32DequantOp,
            node.name + "_values",
            [node, scale],
            node.tensor_meta["shape"],
            TensorDType.Float32,
        )
        graph.replace_as_parent(node, users, decoded)
        decoded.children[:] = users
        node.children[:] = [decoded.name]

    # Quantized consumers use one calibrated product of the accumulator scale
    # and input multiplier, matching the existing ToInt8 Ball contract.
    for node in list(graph.body):
        if not isinstance(node, FP32ToInt8QuantOp):
            continue
        decoded = graph.node_table[node.args[0]]
        if not isinstance(decoded, Int32ToFP32DequantOp):
            continue
        multiplier = parameter(
            graph,
            values,
            node.name + "_accumulator_multiplier",
            values[decoded.args[1]] * values[node.args[1]],
        )
        replace_argument(graph, node, 0, graph.node_table[decoded.args[0]])
        replace_argument(graph, node, 1, multiplier)
        replacement = Int32ToInt8QuantOp()
        replacement.name = node.name
        graph.displace_node(node, replacement)
    pad_quantization(graph, values)

    while True:
        dead = [
            node
            for node in graph.body
            if not node.children and not isinstance(node, OutputOp)
        ]
        if not dead:
            break
        for node in reversed(dead):
            graph.delete_node(
                node, [graph.node_table[parent] for parent in node.parents]
            )

    sort_graph(graph)
    return [values[node.name] for node in graph.params]

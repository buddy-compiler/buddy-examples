import torch

from buddy.compiler.graph.graph import NodeType
from buddy.compiler.graph.operation import (
    MulOp,
    PermuteOp,
    PlaceholderOp,
    SliceOp,
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


def convert(graph, values, op, index, scale):
    """Argument `index` of `op` read as INT8 values with the steps `scale`:
    it converts to INT8 online. Returns the step node."""
    source = graph.node_table[op.args[index]]
    shape = source.tensor_meta["shape"]
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
    return step


def smooth_norms(model, smooth):
    """Smooths the inputs of the channel-mixing fc1s and of the head through
    the LayerNorm that produces them: a Linear converts its whole input with
    one multiplier, so the norm divides its output channel by s (its weight
    and bias; the mean before the head keeps the factor) and the Linear
    weight takes s."""
    pairs = [
        (block.norm2, block.mlp_channels.fc1, f"blocks.{index}.mlp_channels.fc1")
        for index, block in enumerate(model.blocks)
    ]
    pairs.append((model.norm, model.head, "head"))
    with torch.no_grad():
        for norm, linear, name in pairs:
            factor = torch.tensor(smooth[name])
            norm.weight /= factor
            norm.bias /= factor
            linear.weight *= factor


def smatmul_layout(graph, values, weight):
    """The INT8 [N, K] weight stored offline as the B operand of the goban
    smatmul Ball: S = ceil(N / 16) slices of sixteen output channels, K
    zero-padded to a multiple of sixteen and split in G reduction groups of
    g lines, [S, g, G * 16], line l of group j of slice s the sixteen
    channels of column j * g + l. G is the most groups (at most nine: A and
    B take G of the twenty banks of the matmul thread each, the accumulator
    one) that divide K into lines of at most the 1024 of a bank; smaller
    groups let an A tile hold more rows. The matmul thread moves every slice
    in place. Returns the node of the [N, K] weight the graph reads: the
    stored parameter viewed back, a transpose and zero-offset slices."""
    n, k = weight.tensor_meta["shape"]
    padded = (k + 15) // 16 * 16
    groups = max(
        g for g in range(1, 10) if padded % (16 * g) == 0 and padded // g <= 1024
    )
    lines, slices = padded // groups, (n + 15) // 16
    value = torch.nn.functional.pad(
        values[weight.name], (0, padded - k, 0, slices * 16 - n)
    )
    values[weight.name] = (
        value.reshape(slices, 16, groups, lines)
        .permute(0, 3, 2, 1)
        .contiguous()
        .reshape(slices, lines, groups * 16)
    )
    weight.tensor_meta["shape"] = [slices, lines, groups * 16]
    int8 = TensorDType.Int8
    name = weight.name
    node = operation(
        graph, ViewOp, name + "_groups", [weight], [slices, lines, groups, 16], int8
    )
    node.add_argument([slices, lines, groups, 16])
    node = operation(
        graph, PermuteOp, name + "_rows", [node], [slices, 16, groups, lines], int8
    )
    node.add_argument([0, 3, 2, 1])
    node = operation(
        graph, ViewOp, name + "_padded", [node], [slices * 16, padded], int8
    )
    node.add_argument([slices * 16, padded])
    for dim, size, full in ((0, n, slices * 16), (1, k, padded)):
        if size == full:
            continue
        shape = [n, padded] if dim == 0 else [n, k]
        node = operation(graph, SliceOp, f"{name}_slice{dim}", [node], shape, int8)
        for argument in (dim, 0, size):
            node.add_argument(argument)
    return node


def quantize(graph, params, parameter_names, calibration):
    """Every Linear reads INT8 values: its weight with one scale per output
    channel and its input with calibrated scales. A token-mixing Linear reads
    its [C, K] input with one scale per row, a channel of the image: its
    accumulator scale differs per row, so it adds its bias in FP32 after the
    product. Every other Linear reads its input with one scale and adds its
    bias in the INT32 accumulator."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    for op in list(graph.body):
        if not isinstance(op, TransposeMatmulFusedOp):
            continue
        weight = graph.node_table[op.args[1]]
        module = parameter_names[weight.name].rsplit(".", 1)[0]

        # Weights: symmetric in [-127, 127] per output channel, axis 0 of the
        # [N, K] weights.
        value = values[weight.name]
        step = value.abs().amax(1, keepdim=True) / 127
        values[weight.name] = quantize_values(value, 1 / step)
        weight.tensor_meta["dtype"] = TensorDType.Int8
        weight_step = parameter(graph, values, weight.name + "_step", step)
        shape = weight.tensor_meta["shape"]
        weight_codes = smatmul_layout(graph, values, weight)
        weight_values = operation(
            graph,
            Int8ToFP32DequantOp,
            weight.name + "_dequantized",
            [weight_codes, weight_step],
            shape,
            TensorDType.Float32,
        )
        replace_argument(graph, op, 1, weight_values)

        # Input: one scale per row of a token-mixing input, [C, 1]
        # broadcasting over K; one scale for any other tensor, from the
        # calibrated maxima.
        if "mlp_tokens" in module:
            scale = torch.tensor(calibration[module]).reshape(-1, 1) / 127
            convert(graph, values, op, 0, scale)
            continue
        scale = torch.full([1, 1], calibration[module] / 127)
        input_step = convert(graph, values, op, 0, scale)

        # Bias shares the INT32 accumulator's per-output-channel scale; it is
        # the first argument of the add after the product.
        bias_user = graph.node_table[op.children[0]]
        bias = graph.node_table[bias_user.args[0]]
        bias_shape = [value.shape[0]]
        values[bias.name] = quantize_bias(
            values[bias.name], (step * values[input_step.name]).reshape(-1)
        )
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
        replace_argument(graph, bias_user, 0, bias_values)
    sort_graph(graph)
    return [values[node.name] for node in graph.params]

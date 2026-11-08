import torch

from buddy.compiler.graph.graph import NodeType
from buddy.compiler.graph.operation import (
    BatchMatmulOp,
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


def convert(graph, values, op, index, maximum):
    """Argument `index` of `op` read as INT8 values with one scale, from the
    calibrated maximum: it converts to INT8 online. Returns the step node."""
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
    return step


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
    """Every matrix product reads INT8 values. A Linear reads its weight with
    one scale per output channel and its input with one calibrated scale, and
    adds its bias in the INT32 accumulator; an attention product reads both
    operands with one calibrated scale each."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    products = 0
    for op in list(graph.body):
        if isinstance(op, BatchMatmulOp):
            # The attention products in the order the encoder runs them.
            convert(graph, values, op, 0, calibration[f"matmul.{products}.0"])
            convert(graph, values, op, 1, calibration[f"matmul.{products}.1"])
            products += 1
            continue
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

        # Input: one scale for the tensor, from the calibrated maximum.
        input_step = convert(graph, values, op, 0, calibration[module])

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

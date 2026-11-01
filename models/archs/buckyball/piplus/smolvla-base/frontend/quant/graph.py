import torch

from buddy.compiler.graph.graph import NodeType
from buddy.compiler.graph.operation import (
    AddMMOp,
    AddOp,
    BatchMatmulOp,
    ConstantPadNdOp,
    Conv2dOp,
    MatmulOp,
    PermuteOp,
    PlaceholderOp,
    ViewOp,
)
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from buddy.compiler.graph.type import TensorDType
from compiler.frontend.quant import (
    FP32ToMXINT8QuantOp,
    MXINT8ToFP32DequantOp,
    dequantize_mxint8,
    quantize_mxint8,
)


def doubled(flow):
    """The weights read as two MXINT8 terms: those of the MLP of the vision
    encoder, whose MXINT8 error moves the actions most."""
    layers = flow.vlm_with_expert.get_vlm_model().vision_model.encoder.layers
    return {layer.mlp.fc1.weight for layer in layers} | {
        layer.mlp.fc2.weight for layer in layers
    }


def parameter(graph, values, name, value):
    node = PlaceholderOp()
    node.name = name
    node.tensor_meta = {"shape": list(value.shape), "dtype": TensorDType.Int8}
    graph.add_node(node, NodeType.FakeNode)
    values[name] = value
    return node


def operation(graph, op, name, arguments, attributes, shape, dtype):
    node = op()
    node.name = name
    for argument in arguments:
        node.add_argument(argument.name)
        node.add_parent(argument.name)
        argument.add_children(name)
    for attribute in attributes:
        node.add_argument(attribute)
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
        user._parents = [
            node.name if name == source.name else name for name in user.parents
        ]
        source.children.remove(name)
        node.add_children(name)


def packed(shape, axis):
    """The shape of the MXINT8 codes and exponents of values of `shape`."""
    shape = list(shape)
    shape[axis] += shape[axis] // 32
    return shape


def padded(k):
    """K rounded up to the next multiple of 32, the MXINT8 block."""
    return (k + 31) // 32 * 32


def pad(graph, op, index, axis):
    """Argument `index` of `op` padded with zeros along its reduction axis
    `axis` (-1 or -2) to the next multiple of 32."""
    source = graph.node_table[op.args[index]]
    shape = list(source.tensor_meta["shape"])
    extra = padded(shape[axis]) - shape[axis]
    shape[axis] += extra
    # ConstantPadNd lists (before, after) from the last axis backwards.
    widths = [0, extra] if axis == -1 else [0, 0, 0, extra]
    node = operation(
        graph,
        ConstantPadNdOp,
        f"{op.name}_{index}_padded",
        [source],
        [widths, 0.0],
        shape,
        TensorDType.Float32,
    )
    replace_argument(graph, op, index, node)


def mxint8(graph, op, index, axis):
    """Argument `index` of `op` read through MXINT8 along its reduction axis
    `axis`: quantized online and restored to FP32."""
    source = graph.node_table[op.args[index]]
    shape = source.tensor_meta["shape"]
    axis %= len(shape)
    name = f"{op.name}_{index}"
    codes = operation(
        graph,
        FP32ToMXINT8QuantOp,
        name + "_mxint8",
        [source],
        [axis],
        packed(shape, axis),
        TensorDType.Int8,
    )
    restored = operation(
        graph,
        MXINT8ToFP32DequantOp,
        name + "_values",
        [codes],
        [axis],
        shape,
        TensorDType.Float32,
    )
    replace_argument(graph, op, index, restored)


def blockwise(graph, values, weight, transpose):
    """The [N, K] weight, read through `transpose`, packed in MXINT8 along K,
    zero-padded offline to the next multiple of 32."""
    n, k = weight.tensor_meta["shape"]
    value = torch.nn.functional.pad(values[weight.name], (0, padded(k) - k))
    values[weight.name] = quantize_mxint8(value, 1)
    weight.tensor_meta = {"shape": packed(value.shape, 1), "dtype": TensorDType.Int8}
    restored = operation(
        graph,
        MXINT8ToFP32DequantOp,
        weight.name + "_values",
        [weight],
        [1],
        value.shape,
        TensorDType.Float32,
    )
    replace_uses(graph, weight, restored)
    transpose.tensor_meta["shape"] = [value.shape[1], n]
    return value


def residual(graph, values, op, index, weight, value):
    """The second MXINT8 term of a doubled weight: what its first term,
    packed in `weight`, leaves out of the padded `value`, packed offline in
    its own parameter. The product of the same quantized input with it is
    added to `op`."""
    n, k = value.shape
    rest = value - dequantize_mxint8(values[weight.name], 1)
    second = parameter(
        graph, values, weight.name + "_residual", quantize_mxint8(rest, 1)
    )
    second.tensor_meta["shape"] = packed(value.shape, 1)
    f32 = TensorDType.Float32
    restored = operation(
        graph,
        MXINT8ToFP32DequantOp,
        second.name + "_values",
        [second],
        [1],
        value.shape,
        f32,
    )
    transpose = operation(
        graph, PermuteOp, second.name + "_transpose", [restored], [[1, 0]], [k, n], f32
    )
    shape = op.tensor_meta["shape"]
    activation = graph.node_table[op.args[index]]
    product = operation(
        graph, MatmulOp, op.name + "_residual", [activation, transpose], [], shape, f32
    )
    total = operation(graph, AddOp, op.name + "_total", [op, product], [], shape, f32)
    replace_uses(graph, op, total)


def patches(graph, convolution):
    """The patch convolution, whose stride is its kernel, as the product of
    the [P, C * kh * kw] patches of the image with the [F, C * kh * kw]
    weight plus the bias, laid out as the convolution output [1, F, H, W]."""
    image, weight, bias = (graph.node_table[name] for name in convolution.args[:3])
    _, c, height, width = image.tensor_meta["shape"]
    f, _, kh, kw = weight.tensor_meta["shape"]
    h, w = height // kh, width // kw
    name = convolution.name
    f32 = TensorDType.Float32
    blocks = operation(
        graph,
        ViewOp,
        name + "_blocks",
        [image],
        [[c, h, kh, w, kw]],
        [c, h, kh, w, kw],
        f32,
    )
    ordered = operation(
        graph,
        PermuteOp,
        name + "_ordered",
        [blocks],
        [[1, 3, 0, 2, 4]],
        [h, w, c, kh, kw],
        f32,
    )
    rows = operation(
        graph,
        ViewOp,
        name + "_rows",
        [ordered],
        [[h * w, c * kh * kw]],
        [h * w, c * kh * kw],
        f32,
    )
    weight.tensor_meta["shape"] = [f, c * kh * kw]
    transpose = operation(
        graph, PermuteOp, name + "_weight", [weight], [[1, 0]], [c * kh * kw, f], f32
    )
    product = operation(
        graph, AddMMOp, name + "_product", [bias, rows, transpose], [], [h * w, f], f32
    )
    grid = operation(
        graph, ViewOp, name + "_grid", [product], [[1, h, w, f]], [1, h, w, f], f32
    )
    output = operation(
        graph, PermuteOp, name + "_output", [grid], [[0, 3, 1, 2]], [1, f, h, w], f32
    )
    replace_uses(graph, convolution, output)
    graph.delete_node(convolution, [image, weight, bias])
    return product


def quantize(graph, params, doubled):
    """Every matrix product reads MXINT8 operands, blocked along its
    reduction axis K: a Linear its [N, K] weight, packed offline, and its
    input, quantized online; an attention product both operands, online.
    A weight among `doubled` is read as two MXINT8 terms, its MXINT8 value
    and the MXINT8 value of what that leaves out, each packed offline: the
    Linear becomes two MXINT8 products of the same quantized input, added.
    The patch convolution of the vision encoder becomes the Linear of the
    image patches first: a view, a permute and a view gather the patches.

    A product whose K is not a multiple of 32 is zero-padded along K to the
    next multiple: the 720 channels of the action expert to 736, the 241
    keys of the prefix to 256 and the 291 keys of the prefix and the action
    chunk to 320. The weight is stored padded with zero columns; the
    activations are padded online by explicit ConstantPadNd operations
    before their quantization. The zeros add nothing to the products, so
    the padding is exact. Biases stay FP32. Parameters no operation reads
    are dropped."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    doubled = {
        node.name for node, value in zip(graph.params, params) if value in doubled
    }
    for op in list(graph.body):
        if isinstance(op, Conv2dOp):
            weight = graph.node_table[op.args[1]]
            values[weight.name] = values[weight.name].reshape(
                weight.tensor_meta["shape"][0], -1
            )
            op = patches(graph, op)
        if not isinstance(op, (MatmulOp, AddMMOp, BatchMatmulOp)):
            continue
        # x @ W.T (+ b): the input, then the weight read through a transpose.
        index = 1 if isinstance(op, AddMMOp) else 0
        if graph.node_table[op.args[index]].tensor_meta["shape"][-1] % 32:
            pad(graph, op, index, -1)
        mxint8(graph, op, index, -1)
        transpose = graph.node_table[op.args[index + 1]]
        if isinstance(transpose, PermuteOp) and transpose.args[0] in values:
            weight = graph.node_table[transpose.args[0]]
            value = blockwise(graph, values, weight, transpose)
            if weight.name in doubled:
                residual(graph, values, op, index, weight, value)
            continue
        if graph.node_table[op.args[index + 1]].tensor_meta["shape"][-2] % 32:
            pad(graph, op, index + 1, -2)
        mxint8(graph, op, index + 1, -2)
    # A parameter no operation reads is dropped: the weights of the last
    # language model layer past its keys and values, which the prefix
    # computes but nothing reads.
    for node in [node for node in graph.params if not node.children]:
        graph.delete_node(node, [])
    sort_graph(graph)
    return [values[node.name] for node in graph.params]

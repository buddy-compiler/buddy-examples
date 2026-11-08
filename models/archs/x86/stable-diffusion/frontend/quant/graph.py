import torch

from buddy.compiler.graph.graph import NodeType
from buddy.compiler.graph.operation import (
    AbsOp,
    AddOp,
    AmaxOp,
    BatchMatmulOp,
    ClampMinOp,
    Conv2dOp,
    MulOp,
    PlaceholderOp,
    ReciprocalOp,
    SubOp,
    TransposeMatmulFusedOp,
)
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from buddy.compiler.graph.type import TensorDType
from compiler.frontend.quant import (
    FP32ToInt8QuantOp,
    FP32ToMXINT8QuantOp,
    Int8ToFP32DequantOp,
    MXINT8ToFP32DequantOp,
    quantize as quantize_values,
    quantize_mxint8,
)


def parameter(graph, values, name, value):
    node = PlaceholderOp()
    node.name = name
    node.tensor_meta = {"shape": list(value.shape), "dtype": TensorDType.Float32}
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


def int8(graph, values, source, name, axis):
    """`source` as INT8 values with one scale per token: the largest
    magnitude along the reduction axis `axis` over 127, computed online. An
    all-zero token keeps a tiny step: its codes are 0."""
    shape = source.tensor_meta["shape"]
    reduced = list(shape)
    reduced[axis] = 1
    f32 = TensorDType.Float32
    magnitude = operation(graph, AbsOp, name + "_magnitude", [source], [], shape, f32)
    maximum = operation(
        graph,
        AmaxOp,
        name + "_maximum",
        [magnitude],
        [[axis % len(shape)], True],
        reduced,
        f32,
    )
    floor = operation(
        graph, ClampMinOp, name + "_floor", [maximum], [1e-30], reduced, f32
    )
    fraction = parameter(
        graph, values, name + "_fraction", torch.full([1] * len(shape), 1 / 127)
    )
    step = operation(graph, MulOp, name + "_step", [floor, fraction], [], reduced, f32)
    multiplier = operation(
        graph, ReciprocalOp, name + "_multiplier", [step], [], reduced, f32
    )
    codes = operation(
        graph,
        FP32ToInt8QuantOp,
        name + "_codes",
        [source, multiplier],
        [],
        shape,
        TensorDType.Int8,
    )
    return operation(
        graph, Int8ToFP32DequantOp, name + "_values", [codes, step], [], shape, f32
    )


def split(graph, values, op, index, axis):
    """Argument `index` of `op` read as two INT8 terms with one scale per
    token each: its INT8 value plus the INT8 value of what that leaves out."""
    source = graph.node_table[op.args[index]]
    shape = source.tensor_meta["shape"]
    name = f"{op.name}_{index}"
    f32 = TensorDType.Float32
    head = int8(graph, values, source, name + "_head", axis)
    residual = operation(
        graph, SubOp, name + "_residual", [source, head], [], shape, f32
    )
    tail = int8(graph, values, residual, name + "_tail", axis)
    restored = operation(graph, AddOp, name + "_split", [head, tail], [], shape, f32)
    replace_argument(graph, op, index, restored)


def channelwise(graph, values, weight):
    """The weight read as INT8 with one scale per output channel, axis 0, by
    every user."""
    value = values[weight.name]
    step = value.abs().amax(list(range(1, value.ndim)), keepdim=True) / 127
    values[weight.name] = quantize_values(value, 1 / step)
    weight.tensor_meta["dtype"] = TensorDType.Int8
    step = parameter(graph, values, weight.name + "_step", step)
    restored = operation(
        graph,
        Int8ToFP32DequantOp,
        weight.name + "_values",
        [weight, step],
        [],
        value.shape,
        TensorDType.Float32,
    )
    replace_uses(graph, weight, restored)


def blockwise(graph, values, weight):
    """The [N, K] weight packed in MXINT8 along K, read by every user."""
    shape = weight.tensor_meta["shape"]
    values[weight.name] = quantize_mxint8(values[weight.name], 1)
    weight.tensor_meta = {"shape": packed(shape, 1), "dtype": TensorDType.Int8}
    restored = operation(
        graph,
        MXINT8ToFP32DequantOp,
        weight.name + "_values",
        [weight],
        [1],
        shape,
        TensorDType.Float32,
    )
    replace_uses(graph, weight, restored)


def quantize(graph, params):
    """The Linears and the attention products read MXINT8 operands, blocked
    along the reduction axis K: a Linear its [N, K] weight, packed offline,
    and its input, quantized online; an attention product both operands.
    The convolutions read their weights as INT8 with one scale per output
    channel and their inputs as two INT8 terms with one online scale per
    pixel each (one term alone leaves the image at 27.6 dB PSNR). Biases
    stay FP32.

    The products whose K is not a multiple of 32 stay FP32: the products of
    the attention probabilities with the values over the 77 tokens of the
    text encoder and of the UNet's cross-attention."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    for op in list(graph.body):
        if isinstance(op, BatchMatmulOp):
            if graph.node_table[op.args[0]].tensor_meta["shape"][-1] % 32 == 0:
                mxint8(graph, op, 0, -1)
                mxint8(graph, op, 1, -2)
        elif isinstance(op, Conv2dOp):
            # The convolutions run in NHWC: the channels are the last axis.
            split(graph, values, op, 0, -1)
            channelwise(graph, values, graph.node_table[op.args[1]])
        elif isinstance(op, TransposeMatmulFusedOp):
            mxint8(graph, op, 0, -1)
            blockwise(graph, values, graph.node_table[op.args[1]])
    sort_graph(graph)
    return [values[node.name] for node in graph.params]

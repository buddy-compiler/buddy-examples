import torch

from buddy.compiler.graph.graph import NodeType
from buddy.compiler.graph.operation import (
    AbsOp,
    AmaxOp,
    BatchMatmulOp,
    ClampMinOp,
    Conv2dOp,
    MulOp,
    PlaceholderOp,
    ReciprocalOp,
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


def tokenwise(graph, values, op, index, axis):
    """Argument `index` of `op` read as INT8 values with one scale per token:
    the largest magnitude along the reduction axis `axis` over 127, computed
    online. An all-zero token keeps a tiny step: its codes are 0."""
    source = graph.node_table[op.args[index]]
    shape = source.tensor_meta["shape"]
    reduced = list(shape)
    reduced[axis] = 1
    name = f"{op.name}_{index}"
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
    restored = operation(
        graph, Int8ToFP32DequantOp, name + "_values", [codes, step], [], shape, f32
    )
    replace_argument(graph, op, index, restored)


def quantize(graph, params):
    """The Linears and the attention products read MXINT8 operands, blocked
    along the reduction axis K: a Linear its [N, K] weight, packed offline,
    and its input, quantized online; an attention product both operands. The
    two 1-D convolutions read their weights as INT8 with one scale per output
    channel and their inputs as INT8 with one online scale per frame. Biases
    stay FP32.

    The products whose K is not a multiple of 32 stay FP32: the products of
    the attention probabilities with the values over the 1500 audio frames
    (the encoder self-attention and the decoder cross-attention)."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    for op in list(graph.body):
        if isinstance(op, BatchMatmulOp):
            if graph.node_table[op.args[0]].tensor_meta["shape"][-1] % 32 == 0:
                mxint8(graph, op, 0, -1)
                mxint8(graph, op, 1, -2)
            continue
        if isinstance(op, Conv2dOp):
            # The convolutions run in NWC: the channels are the last axis.
            tokenwise(graph, values, op, 0, -1)
            weight = graph.node_table[op.args[1]]
            value = values[weight.name]
            step = value.abs().amax([1, 2], keepdim=True) / 127
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
            continue
        if not isinstance(op, TransposeMatmulFusedOp):
            continue
        mxint8(graph, op, 0, -1)
        weight = graph.node_table[op.args[1]]
        shape = weight.tensor_meta["shape"]
        if weight.children != [op.name]:
            # The output projection shares its weight with the token
            # embedding, which keeps reading the FP32 values: the product
            # reads its own packed copy.
            weight = parameter(
                graph, values, weight.name + "_mxint8", values[weight.name]
            )
            replace_argument(graph, op, 1, weight)
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
        replace_argument(graph, op, 1, restored)
    sort_graph(graph)
    return [values[node.name] for node in graph.params]

from buddy.compiler.graph.graph import NodeType
from buddy.compiler.graph.operation import (
    AbsOp,
    AddOp,
    AddMMOp,
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


def int8(graph, source, name, axis):
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
    step = operation(graph, MulOp, name + "_step", [floor], [1 / 127], reduced, f32)
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


def split(graph, op, index, axis):
    """Argument `index` of `op` read as two INT8 terms with one scale per
    token each: its INT8 value plus the INT8 value of what that leaves out."""
    source = graph.node_table[op.args[index]]
    shape = source.tensor_meta["shape"]
    name = f"{op.name}_{index}"
    f32 = TensorDType.Float32
    head = int8(graph, source, name + "_head", axis)
    residual = operation(
        graph, SubOp, name + "_residual", [source, head], [], shape, f32
    )
    tail = int8(graph, residual, name + "_tail", axis)
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
    """Pack Linear weights in MXINT8 along K when K is divisible by 32;
    otherwise use channelwise INT8, as for convolution weights. Activations
    use two online tokenwise INT8 terms to limit rounding-sensitive drift
    through the denoising steps. Biases stay FP32.

    Attention products with K not divisible by 32 stay FP32: the rotary
    outer products and products over the prefix or action chunk.
    """
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    for op in list(graph.body):
        if isinstance(op, BatchMatmulOp):
            if graph.node_table[op.args[0]].tensor_meta["shape"][-1] % 32 == 0:
                split(graph, op, 0, -1)
                split(graph, op, 1, -2)
            continue
        if isinstance(op, Conv2dOp):
            index, weight, axis = 0, graph.node_table[op.args[1]], 1
        elif isinstance(op, TransposeMatmulFusedOp):
            index, weight, axis = 0, graph.node_table[op.args[1]], -1
        elif isinstance(op, AddMMOp):
            # x @ W.T + b with the weight read through a transpose.
            index, axis = 1, -1
            weight = graph.node_table[graph.node_table[op.args[2]].args[0]]
        else:
            continue
        # A weight read again (the Euler steps, the cameras) is already
        # quantized.
        shared = isinstance(weight, (Int8ToFP32DequantOp, MXINT8ToFP32DequantOp))
        if shared:
            weight = graph.node_table[weight.args[0]]
        k = graph.node_table[op.args[index]].tensor_meta["shape"][axis]
        split(graph, op, index, axis)
        if not shared:
            if isinstance(op, Conv2dOp) or k % 32:
                channelwise(graph, values, weight)
            else:
                blockwise(graph, values, weight)
    sort_graph(graph)
    return [values[node.name] for node in graph.params]

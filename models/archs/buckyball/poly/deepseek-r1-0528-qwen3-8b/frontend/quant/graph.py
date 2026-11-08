from buddy.compiler.graph.operation import (
    AddOp,
    BatchMatmulOp,
    CloneOp,
    DivOp,
    ExpandOp,
    PlaceholderOp,
    SubOp,
    TransposeMatmulFusedOp,
    ViewOp,
)
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from buddy.compiler.graph import NodeType
from buddy.compiler.graph.type import TensorDType
from compiler.frontend.quant import (
    FP32ToMXINT8QuantOp,
    MXINT8ToFP32DequantOp,
    quantize_mxint8,
)


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


def packed(shape, axis):
    """The shape of the MXINT8 codes and exponents of values of `shape`."""
    shape = list(shape)
    shape[axis] += shape[axis] // 32
    return shape


def mxint8(graph, source, name, axis):
    """`source` read through MXINT8: quantized online along `axis` and
    restored to FP32."""
    shape = source.tensor_meta["shape"]
    codes = operation(
        graph,
        FP32ToMXINT8QuantOp,
        f"{name}_mxint8",
        [source],
        [axis],
        packed(shape, axis),
        TensorDType.Int8,
    )
    return operation(
        graph,
        MXINT8ToFP32DequantOp,
        f"{name}_values",
        [codes],
        [axis],
        shape,
        TensorDType.Float32,
    )


def convert(graph, op, index, axis):
    """Argument `index` of `op` read through MXINT8 along `axis`."""
    source = graph.node_table[op.args[index]]
    replace_argument(
        graph, op, index, mxint8(graph, source, f"{op.name}_{index}", axis)
    )


def split(graph, op, index, axis):
    """Argument `index` of `op` read as two MXINT8 terms along `axis`: its
    MXINT8 value plus the MXINT8 value of what that leaves out."""
    source = graph.node_table[op.args[index]]
    shape = source.tensor_meta["shape"]
    name = f"{op.name}_{index}"
    head = mxint8(graph, source, f"{name}_head", axis)
    residual = operation(
        graph, SubOp, f"{name}_residual", [source, head], [], shape, TensorDType.Float32
    )
    tail = mxint8(graph, residual, f"{name}_tail", axis)
    values = operation(
        graph, AddOp, f"{name}_split", [head, tail], [], shape, TensorDType.Float32
    )
    replace_argument(graph, op, index, values)


def probabilities(graph, op):
    """Whether operand 0 of the attention product `op` is the softmax output:
    the division that normalizes it, behind views, expands and clones."""
    source = graph.node_table[op.args[0]]
    while isinstance(source, (ViewOp, ExpandOp, CloneOp)):
        source = graph.node_table[source.args[0]]
    return isinstance(source, DivOp)


def quantize(graph, params):
    """Every matrix product reads MXINT8 operands, blocked along its reduction
    axis K: a Linear its [N, K] weight, packed offline, and its activation,
    quantized online; an attention product both activations. The key of the
    score product q k^T and the probabilities of the product P v are read as
    two MXINT8 terms. A product whose K is not
    a multiple of 32 (the outer product of the rotary angles) stays FP32."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    for op in list(graph.body):
        if isinstance(op, TransposeMatmulFusedOp):
            weight = graph.node_table[op.args[1]]
            shape = weight.tensor_meta["shape"]
            if weight.children != [op.name]:
                # A weight the embedding shares: the Linear reads its own
                # packed copy.
                tied = PlaceholderOp()
                tied.name = weight.name + "_tied"
                tied.tensor_meta = {
                    "shape": list(shape),
                    "dtype": weight.tensor_meta["dtype"],
                }
                graph.add_node(tied, NodeType.FakeNode)
                values[tied.name] = values[weight.name]
                replace_argument(graph, op, 1, tied)
                weight = tied
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
            rank = len(graph.node_table[op.args[0]].tensor_meta["shape"])
            convert(graph, op, 0, rank - 1)
        elif isinstance(op, BatchMatmulOp):
            k = graph.node_table[op.args[0]].tensor_meta["shape"][-1]
            if isinstance(k, int) and k % 32:
                continue
            if probabilities(graph, op):
                split(graph, op, 0, 2)
                convert(graph, op, 1, 1)
            else:
                convert(graph, op, 0, 2)
                split(graph, op, 1, 1)
    sort_graph(graph)
    return [values[node.name] for node in graph.params]

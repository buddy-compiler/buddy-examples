from buddy.compiler.graph.operation import (
    BatchMatmulOp,
    ConstantPadNdOp,
    PlaceholderOp,
    TransposeMatmulFusedOp,
)
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from buddy.compiler.graph import NodeType
from buddy.compiler.graph.sizes import is_symbolic
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


def padded(shape, axis):
    """`shape` with `axis` zero-padded to whole MXINT8 blocks: the token
    count along an attention product is any number."""
    shape = list(shape)
    shape[axis] = (shape[axis] + 31) // 32 * 32
    return shape


def packed(shape, axis):
    """The shape of the MXINT8 codes and exponents of values of `shape`."""
    shape = padded(shape, axis)
    shape[axis] += shape[axis] // 32
    return shape


def mxint8(graph, source, name, axis):
    """`source` read through MXINT8: quantized online along `axis` and
    restored to FP32."""
    shape = source.tensor_meta["shape"]
    if not is_symbolic(shape[axis]) and shape[axis] % 32:
        aligned = padded(shape, axis)
        widths = [0, 0] * (len(shape) - axis - 1) + [0, aligned[axis] - shape[axis]]
        source = operation(
            graph,
            ConstantPadNdOp,
            f"{name}_padded",
            [source],
            [widths, 0.0],
            aligned,
            TensorDType.Float32,
        )
        shape = aligned
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
        padded(shape, axis),
        TensorDType.Float32,
    )


def convert(graph, op, index, axis):
    """Argument `index` of `op` read through MXINT8 along `axis`."""
    source = graph.node_table[op.args[index]]
    replace_argument(
        graph, op, index, mxint8(graph, source, f"{op.name}_{index}", axis)
    )


def quantize(graph, params):
    """Every matrix product reads MXINT8 operands, blocked along its reduction
    axis K: a Linear its [N, K] weight, packed offline, and its activation,
    quantized online; an attention product both activations. Reduction axes
    are zero-padded to complete blocks, including the action head's 1028
    input features."""
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
                padded(shape, 1),
                TensorDType.Float32,
            )
            replace_argument(graph, op, 1, restored)
            rank = len(graph.node_table[op.args[0]].tensor_meta["shape"])
            convert(graph, op, 0, rank - 1)
        elif isinstance(op, BatchMatmulOp):
            k = graph.node_table[op.args[0]].tensor_meta["shape"][-1]
            # A static K off whole blocks stays FP32; the token count K of
            # P V is zero-padded to whole blocks.
            if not is_symbolic(k) and k % 32:
                continue
            convert(graph, op, 1, 1)
            convert(graph, op, 0, 2)
    sort_graph(graph)
    return [values[node.name] for node in graph.params]

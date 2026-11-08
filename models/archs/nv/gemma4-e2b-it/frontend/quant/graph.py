from buddy.compiler.graph import NodeType
from buddy.compiler.graph.operation import (
    BatchMatmulOp,
    EmbeddingOp,
    PlaceholderOp,
    TransposeMatmulFusedOp,
)
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
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


def convert(graph, op, index, axis):
    """Argument `index` of `op` read through MXINT8: quantized online along
    `axis` and restored to FP32."""
    source = graph.node_table[op.args[index]]
    shape = source.tensor_meta["shape"]
    codes = operation(
        graph,
        FP32ToMXINT8QuantOp,
        f"{op.name}_{index}_mxint8",
        [source],
        [axis],
        packed(shape, axis),
        TensorDType.Int8,
    )
    values = operation(
        graph,
        MXINT8ToFP32DequantOp,
        f"{op.name}_{index}_values",
        [codes],
        [axis],
        shape,
        TensorDType.Float32,
    )
    replace_argument(graph, op, index, values)


def quantize(graph, params):
    """Every matrix product reads MXINT8 operands, blocked along its reduction
    axis K: a Linear its [N, K] weight, packed offline, and its activation,
    quantized online; an attention product both activations, the key of
    q k^T and P of P V as one MXINT8 term each (WikiText-2 perplexity +0.6%;
    the key as two terms +0.8%). A product whose K is not a multiple
    of 32 (the outer product of the rotary angles) stays FP32.

    The per-layer embedding table (262144 x 8960, the table no Linear
    shares) is packed to MXINT8 along its rows offline: the gather selects
    packed rows and only they are restored to FP32. The token embedding
    shares its table with lm_head and stays FP32."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    for op in list(graph.body):
        if isinstance(op, EmbeddingOp) and graph.node_table[op.args[0]].children == [
            op.name
        ]:
            table = graph.node_table[op.args[0]]
            shape = table.tensor_meta["shape"]
            values[table.name] = quantize_mxint8(values[table.name], 1)
            table.tensor_meta = {"shape": packed(shape, 1), "dtype": TensorDType.Int8}
            rows = list(op.tensor_meta["shape"])
            op.tensor_meta = {
                "shape": packed(rows, len(rows) - 1),
                "dtype": TensorDType.Int8,
            }
            restored = operation(
                graph,
                MXINT8ToFP32DequantOp,
                op.name + "_values",
                [op],
                [len(rows) - 1],
                rows,
                TensorDType.Float32,
            )
            for user in [
                graph.node_table[name] for name in op.children if name != restored.name
            ]:
                replace_argument(graph, user, user.args.index(op.name), restored)
        elif isinstance(op, TransposeMatmulFusedOp):
            weight = graph.node_table[op.args[1]]
            shape = weight.tensor_meta["shape"]
            if weight.children != [op.name]:
                # A weight the embedding shares (lm_head): the Linear reads
                # its own packed copy, the embedding stays FP32.
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
            if k % 32:
                continue
            convert(graph, op, 0, 2)
            convert(graph, op, 1, 1)
    sort_graph(graph)
    return [values[node.name] for node in graph.params]

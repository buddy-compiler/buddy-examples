"""Q8 attention scores and Q15 probabilities at the existing MXINT8 boundary."""

import torch
import buddy_mlir.ir as ir
from buddy.compiler.graph import NodeType
from buddy.compiler.graph.operation import (
    AmaxOp,
    DivOp,
    ExpOp,
    Op,
    OpType,
    PlaceholderOp,
    SubOp,
    SumDimOp,
)
from buddy.compiler.graph.sizes import mlir_shape
from buddy.compiler.graph.type import TensorDType
from buddy.compiler.graph.transform.quantization.quantize import sort_graph
from frontend.norm import discard, replace

# Offline table generation. The final entry is already zero.
TABLE = torch.round(
    torch.exp(-torch.arange(4096, dtype=torch.float64) / 256) * 32768
).to(torch.int32)


def softmax(values):
    """Host numerical oracle; floating point only represents the boundaries."""
    fixed = torch.round(values.clamp(-4096, 4096) * 256).to(torch.int64)
    indices = (fixed.amax(-1, keepdim=True) - fixed).clamp(0, 4095)
    weights = TABLE[indices].to(torch.int64)
    total = weights.sum(-1, keepdim=True)
    probability = torch.div(weights * 32768 + total // 2, total, rounding_mode="floor")
    return probability.to(values.dtype) / 32768


class IntegerSoftmaxOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ReduceType


def integer_softmax_op(node, symbol_table):
    return ir.Operation.create(
        "frontend.integer_softmax",
        results=[
            ir.RankedTensorType.get(
                mlir_shape(node.tensor_meta["shape"]), ir.F32Type.get()
            )
        ],
        operands=[symbol_table[(name, 0)] for name in node.args],
    ).result


ops_registry = {"IntegerSoftmaxOp": integer_softmax_op}


def integer_softmax(graph, params):
    """Replace attention softmax expressions before device partitioning."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    table = PlaceholderOp()
    table.name = "integer_softmax_table"
    table.tensor_meta = {"shape": [4096], "dtype": TensorDType.Int32}
    graph.add_node(table, NodeType.FakeNode)
    values[table.name] = TABLE
    count = 0
    for exponential in [node for node in graph.body if isinstance(node, ExpOp)]:
        difference = graph.node_table[exponential.args[0]]
        if not isinstance(difference, SubOp):
            continue
        maximum = graph.node_table[difference.args[1]]
        source = graph.node_table[difference.args[0]]
        if len(source.tensor_meta["shape"]) < 3:
            continue
        if not isinstance(maximum, AmaxOp) or maximum.args != [source.name, [-1], True]:
            raise ValueError(f"{exponential.name}: expected last-axis softmax maximum")
        (summation,) = [
            graph.node_table[name]
            for name in exponential.children
            if isinstance(graph.node_table[name], SumDimOp)
        ]
        (division,) = [
            graph.node_table[name]
            for name in exponential.children
            if isinstance(graph.node_table[name], DivOp)
        ]
        if summation.args != [exponential.name, [-1], True] or division.args != [
            exponential.name,
            summation.name,
        ]:
            raise ValueError(f"{exponential.name}: expected softmax sum and division")
        parents = replace(graph, division, IntegerSoftmaxOp(), [source, table], [])
        discard(graph, parents)
        count += 1
    if not count:
        raise ValueError("Laya graph contains no attention softmax expressions")
    sort_graph(graph)
    print(f"Mapped {count} attention softmax expressions to Q8/Q15 integer lookup")
    return [values[node.name] for node in graph.params]

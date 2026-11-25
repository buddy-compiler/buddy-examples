from buddy.compiler.graph.operation import (
    AddOp,
    IndexPutOp,
    IotaOp,
    MatmulOp,
    PermuteOp,
)
from buddy.compiler.graph.transform.fuse_ops import transpose_matmul_fusion


def permute(graph):
    """x @ W.T of a Linear becomes one matmul with a transposed right operand,
    so every weight keeps the [out, in] layout of the checkpoint."""
    for node in graph.body:
        if not isinstance(node, IndexPutOp):
            continue
        if len(node.args[1]) != 3 or node.args[1][:2] != [None, None]:
            raise ValueError("KV updates must index only the token axis")
        positions = graph.node_table[node.args[1][2]]
        if not isinstance(positions, AddOp):
            raise ValueError("KV positions must be start plus consecutive tokens")
        ranges = [
            graph.node_table[name]
            for name in positions.parents
            if isinstance(graph.node_table[name], IotaOp)
        ]
        if (
            len(ranges) != 1
            or ranges[0].kwargs["start"] != 0
            or ranges[0].kwargs["step"] != 1
        ):
            raise ValueError("KV positions must have unit stride")
    for matmul in list(graph.body):
        if not isinstance(matmul, MatmulOp):
            continue
        weight = graph.node_table[matmul.args[1]]
        if not isinstance(weight, PermuteOp) or list(weight.args[1]) != [1, 0]:
            continue
        parents = [graph.node_table[name] for name in matmul.parents]
        transpose_matmul_fusion(
            graph, matmul, weight, parents, "transpose_matmul_fusion"
        )

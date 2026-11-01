from buddy.compiler.graph.operation import MatmulOp, PermuteOp
from buddy.compiler.graph.transform.fuse_ops import (
    decompose_addmm_to_mm_add,
    transpose_matmul_fusion,
)


def permute(graph):
    """x @ W.T of a Linear becomes one matmul with a transposed right operand,
    so every weight keeps the [out, in] layout of the checkpoint."""
    decompose_addmm_to_mm_add(graph)
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

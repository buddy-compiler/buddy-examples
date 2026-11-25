import torch

from buddy.compiler.graph.graph import NodeType
from buddy.compiler.graph.operation import MatmulOp, PermuteOp, PlaceholderOp
from buddy.compiler.graph.type import TensorDType


def weights(graph, params):
    """The weight of every Linear as the mesh reads it: [N, K] in FP16, where
    the checkpoint keeps [K, N] in FP32 and the graph transposes it at every
    run. The product reads the parameter in place. A weight something else
    reads too (the embedding table, which is the weight of the last Linear)
    stays, and the Linear gets its own copy."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    for matmul in list(graph.body):
        if not isinstance(matmul, MatmulOp):
            continue
        permute = graph.node_table[matmul.args[1]]
        if not isinstance(permute, PermuteOp) or list(permute.args[1]) != [1, 0]:
            continue
        if permute.children != [matmul.name]:
            raise ValueError(f"{permute.name} feeds more than {matmul.name}")
        weight = graph.node_table[permute.args[0]]
        if not isinstance(weight, PlaceholderOp):
            raise ValueError(f"{matmul.name} multiplies by a computed weight")
        value = values[weight.name].detach().t().contiguous().to(torch.float16)
        meta = {"shape": list(value.shape), "dtype": TensorDType.Float16}
        weight.children.remove(permute.name)
        if weight.children:
            source = weight
            weight = PlaceholderOp()
            weight.name = source.name + "_linear"
            graph.add_node(weight, NodeType.FakeNode)
        weight.tensor_meta = meta
        values[weight.name] = value
        weight.add_children(matmul.name)
        matmul.args[1] = weight.name
        matmul.parents[matmul.parents.index(permute.name)] = weight.name
        permute.children.remove(matmul.name)
        graph.delete_node(permute, [])
    return [values[node.name] for node in graph.params]

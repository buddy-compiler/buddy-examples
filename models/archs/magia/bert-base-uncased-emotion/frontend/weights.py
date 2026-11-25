import torch

from buddy.compiler.graph.operation import AddMMOp, PermuteOp, PlaceholderOp
from buddy.compiler.graph.type import TensorDType


def weights(graph, params):
    """The weight of every Linear as the mesh reads it: [N, K] in FP16, where
    the checkpoint keeps [K, N] in FP32 and the graph transposes it at every
    run. The product reads the parameter in place."""
    values = dict(zip((node.name for node in graph.params), params, strict=True))
    for matmul in list(graph.body):
        if not isinstance(matmul, AddMMOp):
            continue
        permute = graph.node_table[matmul.args[2]]
        if not isinstance(permute, PermuteOp) or list(permute.args[1]) != [1, 0]:
            continue
        if permute.children != [matmul.name]:
            raise ValueError(f"{permute.name} feeds more than {matmul.name}")
        weight = graph.node_table[permute.args[0]]
        if not isinstance(weight, PlaceholderOp):
            raise ValueError(f"{matmul.name} multiplies by a computed weight")
        value = values[weight.name].detach().t().contiguous().to(torch.float16)
        meta = {"shape": list(value.shape), "dtype": TensorDType.Float16}
        if weight.children != [permute.name]:
            raise ValueError(f"{weight.name} feeds more than {matmul.name}")
        weight.children.remove(permute.name)
        weight.tensor_meta = meta
        values[weight.name] = value
        weight.add_children(matmul.name)
        matmul.args[2] = weight.name
        matmul.parents[matmul.parents.index(permute.name)] = weight.name
        permute.children.remove(matmul.name)
        graph.delete_node(permute, [])
    return [values[node.name] for node in graph.params]

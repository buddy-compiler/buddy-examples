from stack.compiler.quant import mxfp8_cache as cache
from copy import deepcopy

from buddy.compiler.graph.operation import EmbeddingOp, MatmulOp, PermuteOp, TOp
from stack.compiler.quant.mxfp8_graph import quantize_graph


def quantize(graph, params, output, name, packer, bank_bytes):
    parameters = list(graph.params)
    runtime_inputs = [graph._body[index] for index in graph._inputs]
    names = [node.name for node in parameters]
    positions = {node.name: index for index, node in enumerate(parameters)}
    transposes = {}
    for node in graph.body:
        if not isinstance(node, MatmulOp):
            continue
        weight = graph.node_table[node.args[1]]
        if not (
            isinstance(weight, TOp)
            or isinstance(weight, PermuteOp)
            and list(weight.args[1]) == [1, 0]
        ):
            raise ValueError(
                f"Gemma linear requires a transposed parameter: {node.name}"
            )
        parameter = graph.node_table[weight._parents[0]]
        if parameter.name not in positions:
            raise ValueError(f"Gemma linear weight is not a parameter: {node.name}")
        transposes.setdefault(parameter.name, set()).add(weight.name)
    selected = set()
    for original, uses in transposes.items():
        parameter = graph.node_table[original]
        if set(parameter._children) != uses:
            # A tied embedding and linear projection use distinct packed layouts.
            clone = deepcopy(parameter)
            clone._name = original + "_mxfp8"
            clone._children = sorted(uses)
            graph._body.insert(graph._body.index(parameter) + 1, clone)
            graph.node_table[clone.name] = clone
            parameters.append(clone)
            params.append(params[positions[original]])
            names.append(clone.name)
            for transpose_name in uses:
                transpose = graph.node_table[transpose_name]
                transpose._parents[0] = clone.name
                transpose._arguments[0] = clone.name
                parameter._children.remove(transpose_name)
            parameter = clone
        selected.add(parameter.name)
    graph._fake_params = [graph._body.index(node) for node in parameters]
    graph._inputs = [graph._body.index(node) for node in runtime_inputs]
    embeddings = {node.args[0] for node in graph.body if isinstance(node, EmbeddingOp)}
    if not selected:
        raise ValueError("Gemma graph has no linear weights")
    return quantize_graph(
        graph,
        params,
        names,
        output,
        name,
        packer,
        weights=selected,
        embeddings=embeddings,
        bank_bytes=bank_bytes,
        rows_hint=512,
    )

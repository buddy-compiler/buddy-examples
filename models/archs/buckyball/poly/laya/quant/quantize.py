from buddy.compiler.graph.operation import EmbeddingOp
from stack.compiler.quant.mxfp8_graph import quantize_graph
from stack.compiler.quant.mxfp8_embedding import quant_embedding
from examples.balls.mxmm.compiler.python.layout import bank_bytes

FORMAT = "mxfp8_e4m3"
PREPARE = None
STAGES = {"embedding", "attention", "ffn", "typed", "scorer", "action"}


def apply(graph, params, names, output, name, packer, stage, samples):
    parameters = {node.name: parameter for node, parameter in zip(graph.params, names)}
    embeddings = {
        parameters[node.args[0]] for node in graph.body if isinstance(node, EmbeddingOp)
    }
    weights = {
        key
        for key, value in zip(names, params)
        if key.endswith("weight") and value.ndim == 2 and key not in embeddings
    }
    if embeddings and not weights:
        quant_embedding(
            graph, params, names, output, name, packer, embeddings=embeddings
        )
        return
    quantize_graph(
        graph,
        params,
        names,
        output,
        name,
        packer,
        weights=weights,
        embeddings=embeddings,
        bank_bytes=bank_bytes(packer.parent.parent, graph.packing_target),
        rows_hint=next(iter(samples[0].values())).shape[-2],
    )

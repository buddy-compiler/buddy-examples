from buddy.compiler.graph.operation import EmbeddingOp
from stack.compiler.quant.mxfp8_graph import quantize_graph
from stack.compiler.quant.mxfp8_embedding import quant_embedding
from stack.compiler.quant import mxfp8_cache as cache
from examples.balls.mxmm.compiler.python.layout import bank_bytes


def quantize(graph, params, names, output, name, packer, *, kind):
    if kind not in {"embedding", "attention", "ffn", "output"}:
        raise ValueError(f"Unknown model stage: {kind}")
    if kind == "embedding":
        parameters = {
            node.name: parameter for node, parameter in zip(graph.params, names)
        }
        embeddings = {
            parameters[node.args[0]]
            for node in graph.body
            if isinstance(node, EmbeddingOp)
        }
        quant_embedding(
            graph, params, names, output, name, packer, embeddings=embeddings
        )
        return
    weights = {
        parameter for parameter, tensor in zip(names, params) if tensor.ndim == 2
    }
    quantize_graph(
        graph,
        params,
        names,
        output,
        name,
        packer,
        weights=weights,
        embeddings=set(),
        bank_bytes=bank_bytes(
            packer.parent.parent,
            "attention" if kind in {"embedding", "attention"} else "ffn",
        ),
        rows_hint=graph.packing_rows,
    )

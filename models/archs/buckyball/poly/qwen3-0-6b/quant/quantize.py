from stack.compiler.quant.mxfp8_graph import quantize_graph
from examples.balls.mxmm.compiler.python.layout import bank_bytes


def quantize(graph, params, names, output, name, packer, *, kind):
    if kind not in {"embedding", "attention", "ffn", "output"}:
        raise ValueError(f"Unknown model stage: {kind}")
    # Embeddings and norm/rotary parameters remain FP32; linear matrices use MXFP8.
    weights = set() if kind == "embedding" else {
        name for name, param in zip(names, params) if param.ndim == 2
    }
    quantize_graph(graph, params, names, output, name, packer, weights=weights,
                   bank_bytes=bank_bytes(packer.parent.parent, "attention" if kind in {"embedding", "attention"} else "ffn"),
                   rows_hint=graph.packing_rows)

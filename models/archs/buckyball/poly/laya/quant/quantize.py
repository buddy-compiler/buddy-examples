from stack.compiler.quant.mxfp8_graph import quantize_graph
from examples.balls.mxmm.compiler.python.layout import bank_bytes

FORMAT = "mxfp8_e4m3"
PREPARE = None
STAGES = {"attention", "ffn"}


def apply(graph, params, names, output, name, packer, stage, samples):
    weights = {
        key
        for key, value in zip(names, params)
        if key.endswith("weight") and value.ndim == 2
    }
    quantize_graph(
        graph,
        params,
        names,
        output,
        name,
        packer,
        weights=weights,
        bank_bytes=bank_bytes(packer.parent.parent, graph.packing_target),
        rows_hint=samples[0]["hidden"].shape[-2],
    )

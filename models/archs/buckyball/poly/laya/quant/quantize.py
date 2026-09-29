from stack.compiler.quant.mxfp8_graph import quantize_graph
from ..permute.layout import reorder, TILE_ROWS, TILE_K

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
        reorder=reorder,
        tile_rows=TILE_ROWS,
        tile_k=TILE_K,
    )

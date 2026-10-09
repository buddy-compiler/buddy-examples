import numpy as np
from buddy.compiler.graph.transform.layout.matrix import matrix_tiles

TILE_ROWS = 16
TILE_K = 512


def reorder(codes, scales):
    rows, width = codes.shape
    if width % 32 or scales.shape != (rows, width // 32):
        raise ValueError("MXFP8 matrix and block scales disagree")
    padded_rows = (rows + TILE_ROWS - 1) // TILE_ROWS * TILE_ROWS
    padded_k = (width + TILE_K - 1) // TILE_K * TILE_K
    codes = np.pad(codes, ((0, padded_rows - rows), (0, padded_k - width)))
    scales = np.pad(scales, ((0, padded_rows - rows), (0, (padded_k - width) // 32)), constant_values=127)
    # Each N/K tile stores its E4M3 code plane followed by its E8M0 scale plane.
    codes = matrix_tiles(codes, TILE_ROWS, TILE_K).reshape(-1, TILE_ROWS * TILE_K)
    scales = matrix_tiles(scales, TILE_ROWS, TILE_K // 32).reshape(-1, TILE_ROWS * TILE_K // 32)
    return np.concatenate((codes, scales), axis=1).ravel()


def apply(graph, *, kind):
    if kind == "attention":
        from examples.balls.mxmm.compiler.python import fp32
        fp32.apply(graph, fused=False)

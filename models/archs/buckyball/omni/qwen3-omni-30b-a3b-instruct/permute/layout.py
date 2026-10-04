import numpy as np
from buddy.compiler.graph.transform.layout.matrix import matrix_tiles

TILE_ROWS = 16
TILE_K = 512


def reorder(codes, scales):
    rows, width = codes.shape
    if width % 32 or scales.shape != (rows, width // 32):
        raise ValueError("MXFP8 matrix and block scales disagree")
    padded_rows = (rows + TILE_ROWS - 1) // TILE_ROWS * TILE_ROWS
    tile_k = min(width, TILE_K)
    padded_k = (width + tile_k - 1) // tile_k * tile_k
    codes = np.pad(codes, ((0, padded_rows - rows), (0, padded_k - width)))
    scales = np.pad(
        scales,
        ((0, padded_rows - rows), (0, (padded_k - width) // 32)),
        constant_values=127,
    )
    codes = matrix_tiles(codes, TILE_ROWS, tile_k).reshape(-1, TILE_ROWS * tile_k)
    scales = matrix_tiles(scales, TILE_ROWS, tile_k // 32).reshape(
        -1, TILE_ROWS * tile_k // 32
    )
    return np.concatenate((codes, scales), axis=1).ravel()

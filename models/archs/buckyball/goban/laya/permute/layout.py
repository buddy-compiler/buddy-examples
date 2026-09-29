import numpy as np


def reorder(weight, scales, depthwise):
    if weight.ndim != 2 or depthwise:
        raise ValueError("Laya weights must be matrices")
    packed = weight.T.copy()
    scales = np.asarray(scales, dtype=np.float32).reshape(-1)
    scales = np.pad(scales, (0, (-scales.size) % 16), constant_values=1.0)
    return packed, scales

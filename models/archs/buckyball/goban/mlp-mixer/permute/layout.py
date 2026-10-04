import numpy as np


def reorder(weight, scales, depthwise):
    if weight.ndim != 2 or depthwise:
        raise ValueError("Mixer weights must be matrices")
    scales = np.asarray(scales, dtype=np.float32).reshape(-1)
    return weight.T.copy(), np.pad(scales, (0, (-scales.size) % 16), constant_values=1.0)

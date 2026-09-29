import numpy as np
from buddy.compiler.graph.transform.layout.convolution import block_conv


def reorder(weight, scales, depthwise):
    if depthwise:
        raise ValueError("This design has no depthwise convolution")
    if weight.ndim == 4:
        # [O, I, H, W] -> [O/16, I, padded(H*W), 16].
        packed = block_conv(weight, output_lanes=16, spatial_alignment=16)
    elif weight.ndim == 2:
        # Linear weights: [O, I] -> [I, O].
        packed = weight.T.copy()
    else:
        raise ValueError("Expected convolution or linear weights")
    scales = np.asarray(scales, dtype=np.float32).reshape(-1)
    scales = np.pad(scales, (0, (-scales.size) % 16), constant_values=1.0)
    return packed, scales

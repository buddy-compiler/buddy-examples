import numpy as np
import torch
from torch.nn import functional as F
from buddy.compiler.graph.transform.layout.convolution import block_conv


def projection_weights(weight):
    # Native W-then-H unfolding produces [N, C, OH, OW, KW, KH].
    # Match each channel's flattened [KW, KH] offline, keeping activations
    # in their native order instead of transposing a six-dimensional tensor.
    kernels = [weight[:, channel].transpose(1, 2).flatten(1).contiguous()
               for channel in range(weight.shape[1])]
    return [F.pad(kernel, (0, (-kernel.shape[1]) % 16)) for kernel in kernels]


def padded_linear_parameters(weight, bias):
    # Extra output channels are discarded. Duplicates keep quantization's
    # nonzero-scale contract without changing any logical output channel.
    padding = (-weight.shape[0]) % 16
    return (torch.cat((weight, weight[:padding])),
            torch.cat((bias, bias[:padding])))


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


def pool_result(node, symbols):
    from buddy.compiler.ops import tosa as ops
    # Keep the projection/pool boundary physically NHWC for the next Conv.
    return ops.maxpool2d_op(node, symbols).result


def projection_output(node, symbols):
    from buddy_mlir import ir
    value = symbols[(str(node.args[0]), 0)]
    n, c, h, w = node.tensor_meta["shape"]
    if list(node.args[1]) != [0, 3, 1, 2] or list(ir.RankedTensorType(value.type).shape) != [n, h, w, c]:
        raise ValueError("AlexNet projection output requires the NHWC layout")
    return value

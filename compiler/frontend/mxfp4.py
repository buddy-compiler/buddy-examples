"""MXFP4 (OCP MX, FP4 E2M1 elements): a packed weight format, its decode
operation frontend.mxfp4_to_fp32 and its offline quantization.

Every 32 consecutive values along an axis share one E8M0 exponent
e = floor(log2(max |x|)) - 2 (the largest E2M1 value, 6, is 1.5 * 2^2); each
value is the E2M1 code nearest x / 2^e, ties to the even code, magnitudes past
6 clamped to 6. A packed MXFP4 tensor has, along that axis, the K / 2 code
bytes (value 2i in the low nibble of byte i, value 2i + 1 in the high one),
then the K / 32 exponent bytes (e + 127): K + K / 32 values take
K / 2 + K / 32 bytes, half the K codes of MXINT8.

An E2M1 value times two is an integer in [-12, 12], so a block's dot product
with INT8 codes stays exact in INT32 with its exponent one lower."""

import buddy_mlir.ir as ir
import torch
import torch.nn.functional as F

from buddy.compiler.graph.operation import Op, OpType
from buddy.compiler.graph.sizes import mlir_shape


class MXFP4ToFP32DequantOp(Op):
    """frontend.mxfp4_to_fp32(packed, axis): the values of a packed MXFP4 tensor."""

    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType


def mxfp4_to_fp32_dequant_op(node, symbol_table):
    return ir.Operation.create(
        "frontend.mxfp4_to_fp32",
        results=[
            ir.RankedTensorType.get(
                mlir_shape(node.tensor_meta["shape"]), ir.F32Type.get()
            )
        ],
        operands=[symbol_table[(str(node.args[0]), 0)]],
        attributes={
            "axis": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), node.args[1])
        },
    ).result


# Merged into the importer's registry beside compiler.frontend.quant's.
ops_registry = {"MXFP4ToFP32DequantOp": mxfp4_to_fp32_dequant_op}

# The magnitudes of the E2M1 codes 0..7; code bit 3 is the sign.
VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def quantize_mxfp4(values, axis):
    """The packed MXFP4 tensor of `values`, the axis a multiple of 32."""
    x = values.double().movedim(axis, -1).contiguous()
    if x.shape[-1] % 32:
        raise ValueError("MXFP4 needs complete 32-value blocks")
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    maximum = blocks.abs().amax(-1, keepdim=True)
    exponent = torch.where(maximum == 0, 0, torch.frexp(maximum)[1] - 3).clamp(min=-127)
    scaled = torch.ldexp(blocks, -exponent).abs().clamp(max=6.0)
    grid = torch.tensor(VALUES, dtype=torch.float64)
    # Nearest magnitude; a tie goes to the even code (an even mantissa bit).
    cost = (scaled[..., None] - grid).abs() * 2 + (torch.arange(8) % 2) * 1e-9
    magnitude = cost.argmin(-1)
    sign = ((blocks < 0) & (magnitude != 0)).long() << 3
    codes = (magnitude | sign).reshape(x.shape)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    scales = (exponent[..., 0] + 127).to(torch.uint8)
    return torch.cat([packed, scales], -1).view(torch.int8).movedim(-1, axis)


def dequantize_mxfp4(packed, axis):
    """The K values of a packed MXFP4 tensor along `axis`."""
    p = packed.movedim(axis, -1).contiguous().view(torch.uint8).long()
    k = p.shape[-1] * 32 // 17
    codes = torch.stack([p[..., : k // 2] & 15, p[..., : k // 2] >> 4], -1).flatten(-2)
    grid = torch.tensor(VALUES, dtype=torch.float64)
    value = grid[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    exponent = (p[..., k // 2 :] - 127).repeat_interleave(32, -1)
    return torch.ldexp(value, exponent).float().movedim(-1, axis)


def mxfp4(x, axis):
    """x zero-padded along `axis` to a multiple of 32 and read through MXFP4."""
    axis %= x.ndim
    x = F.pad(x, (0, 0) * (x.ndim - 1 - axis) + (0, -x.shape[axis] % 32))
    return dequantize_mxfp4(quantize_mxfp4(x, axis), axis)

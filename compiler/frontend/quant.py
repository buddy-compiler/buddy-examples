"""Symmetric INT8 quantization: the graph operations a frontend inserts and
their lowering.

quantize(x, multiplier) = clamp(round_half_even(x * multiplier), -128, 127)
dequantize(q, step)     = q * step

multiplier and step are FP32 tensors that broadcast against x.

ops_registry lowers each operation to one operation of the frontend dialect
(compiler/midend/Dialect/Frontend); a design runs it on its hardware, or
lower-frontend-fake-op computes it in software. ref_registry
lowers them to their computation, to evaluate the accuracy of a quantized graph
on the host.

MXINT8 (OCP MX, INT8 elements): every 32 consecutive values along an axis
share one E8M0 exponent e = floor(log2(max |x|)) - 6, and each value is the
code clamp(round_half_even(x / 2^e), -127, 127). A packed MXINT8 tensor has
the shape of its values with that axis of length K + K / 32: the K INT8 codes,
then the K / 32 exponent bytes (e + 127). FP32ToMXINT8QuantOp(x, axis) packs x;
MXINT8ToFP32DequantOp(m, axis) unpacks it. A K that is no multiple of 32 (a
dynamic token count) is zero-padded to Kp = ceil(K / 32) * 32: a zero changes
neither the maximum of its block nor any product, so the packed axis holds Kp
codes and Kp / 32 exponents and unpacks to the Kp values, zeros after K.
"""

import buddy_mlir.ir as ir
import torch
from buddy_mlir.dialects import tosa

from buddy.compiler.graph.operation import Op, OpType
from buddy.compiler.graph.sizes import mlir_shape


class FP32ToInt8QuantOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType


class Int8ToFP32DequantOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType


class Int32ToFP32DequantOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType


class FP32ToMXINT8QuantOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType


class MXINT8ToFP32DequantOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType


def quantize(values, multiplier):
    """The quantization of FP32ToInt8QuantOp, for offline values."""
    return torch.clamp(torch.round(values * multiplier), -128, 127).to(torch.int8)


def quantize_bias(values, step):
    """Encode bias in the INT32 accumulation domain, rounding ties to even."""
    codes = torch.round(values / step)
    if not torch.all(torch.isfinite(codes) & (codes >= -(2**31)) & (codes < 2**31)):
        raise ValueError("Quantized bias does not fit INT32")
    return codes.to(torch.int32)


def quantize_mxint8(values, axis):
    """FP32ToMXINT8QuantOp on offline values: the packed INT8 MXINT8 tensor."""
    x = values.double().movedim(axis, -1)
    x = torch.nn.functional.pad(x, (0, -x.shape[-1] % 32)).contiguous()
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    maximum = blocks.abs().amax(-1, keepdim=True)
    exponent = torch.where(maximum == 0, 0, torch.frexp(maximum)[1] - 7).clamp(min=-127)
    codes = torch.clamp(torch.round(torch.ldexp(blocks, -exponent)), -127, 127)
    codes = codes.reshape(x.shape).to(torch.int8)
    scales = (exponent[..., 0] + 127).to(torch.uint8).view(torch.int8)
    return torch.cat([codes, scales], -1).movedim(-1, axis)


def dequantize_mxint8(packed, axis):
    """MXINT8ToFP32DequantOp on offline values: the Kp values."""
    p = packed.movedim(axis, -1).contiguous()
    k = p.shape[-1] * 32 // 33
    exponent = p[..., k:].view(torch.uint8).long() - 127
    value = torch.ldexp(p[..., :k].double(), exponent.repeat_interleave(32, -1))
    return value.float().movedim(-1, axis)


def shift():
    i8 = ir.IntegerType.get_signless(8)
    zero = ir.DenseElementsAttr.get_splat(
        ir.RankedTensorType.get([1], i8), ir.IntegerAttr.get(i8, 0)
    )
    return tosa.ConstOp(zero).results[0]


def fp32_to_int8_quant_reference(node, symbol_table):
    values = symbol_table[(str(node.args[0]), 0)]
    multiplier = symbol_table[(str(node.args[1]), 0)]
    shape = mlir_shape(node.tensor_meta["shape"])
    scaled = tosa.MulOp(
        ir.RankedTensorType.get(shape, ir.F32Type.get()),
        values,
        multiplier,
        shift(),
    ).result
    return tosa.CastOp(
        ir.RankedTensorType.get(shape, ir.IntegerType.get_signless(8)), scaled
    )


def int8_to_fp32_dequant_reference(node, symbol_table):
    codes = symbol_table[(str(node.args[0]), 0)]
    step = symbol_table[(str(node.args[1]), 0)]
    f32 = ir.RankedTensorType.get(
        mlir_shape(node.tensor_meta["shape"]), ir.F32Type.get()
    )
    values = tosa.CastOp(f32, codes).result
    return tosa.MulOp(f32, values, step, shift())


def int32_to_fp32_dequant_reference(node, symbol_table):
    codes = symbol_table[(str(node.args[0]), 0)]
    step = symbol_table[(str(node.args[1]), 0)]
    f32 = ir.RankedTensorType.get(
        mlir_shape(node.tensor_meta["shape"]), ir.F32Type.get()
    )
    values = tosa.CastOp(f32, codes).result
    return tosa.MulOp(f32, values, step, shift())


def fp32_to_int8_quant_op(node, symbol_table):
    i8 = ir.IntegerType.get_signless(8)
    return ir.Operation.create(
        "frontend.fp32_to_int8",
        results=[ir.RankedTensorType.get(mlir_shape(node.tensor_meta["shape"]), i8)],
        operands=[symbol_table[(str(argument), 0)] for argument in node.args],
    ).result


def int8_to_fp32_dequant_op(node, symbol_table):
    f32 = ir.F32Type.get()
    return ir.Operation.create(
        "frontend.int8_to_fp32",
        results=[ir.RankedTensorType.get(mlir_shape(node.tensor_meta["shape"]), f32)],
        operands=[symbol_table[(str(argument), 0)] for argument in node.args],
    ).result


def int32_to_fp32_dequant_op(node, symbol_table):
    f32 = ir.F32Type.get()
    return ir.Operation.create(
        "frontend.int32_to_fp32",
        results=[ir.RankedTensorType.get(mlir_shape(node.tensor_meta["shape"]), f32)],
        operands=[symbol_table[(str(argument), 0)] for argument in node.args],
    ).result


def fp32_to_mxint8_quant_op(node, symbol_table):
    i8 = ir.IntegerType.get_signless(8)
    return ir.Operation.create(
        "frontend.fp32_to_mxint8",
        results=[ir.RankedTensorType.get(mlir_shape(node.tensor_meta["shape"]), i8)],
        operands=[symbol_table[(str(node.args[0]), 0)]],
        attributes={
            "axis": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), node.args[1])
        },
    ).result


def mxint8_to_fp32_dequant_op(node, symbol_table):
    f32 = ir.F32Type.get()
    return ir.Operation.create(
        "frontend.mxint8_to_fp32",
        results=[ir.RankedTensorType.get(mlir_shape(node.tensor_meta["shape"]), f32)],
        operands=[symbol_table[(str(node.args[0]), 0)]],
        attributes={
            "axis": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), node.args[1])
        },
    ).result


ops_registry = {
    "FP32ToInt8QuantOp": fp32_to_int8_quant_op,
    "Int8ToFP32DequantOp": int8_to_fp32_dequant_op,
    "Int32ToFP32DequantOp": int32_to_fp32_dequant_op,
    "FP32ToMXINT8QuantOp": fp32_to_mxint8_quant_op,
    "MXINT8ToFP32DequantOp": mxint8_to_fp32_dequant_op,
}

ref_registry = {
    "FP32ToInt8QuantOp": fp32_to_int8_quant_reference,
    "Int8ToFP32DequantOp": int8_to_fp32_dequant_reference,
    "Int32ToFP32DequantOp": int32_to_fp32_dequant_reference,
}

"""Symmetric INT8 quantization: the graph operations a frontend inserts and
their lowering.

quantize(x, multiplier) = clamp(round_half_even(x * multiplier), -128, 127)
dequantize(q, step)     = q * step

multiplier and step are FP32 tensors that broadcast against x.

ops_registry lowers each operation to one opaque frontend operation; the
midend decides whether software or a Ball implements it. ref_registry
lowers them to their computation, to evaluate the accuracy of a quantized graph
on the host.
"""

import buddy_mlir.ir as ir
import torch
from buddy_mlir.dialects import tosa

from buddy.compiler.graph.operation import Op, OpType


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


def quantize(values, multiplier):
    """The quantization of FP32ToInt8QuantOp, for offline values."""
    return torch.clamp(torch.round(values * multiplier), -128, 127).to(torch.int8)


def quantize_bias(values, step):
    """Encode bias in the INT32 accumulation domain, rounding ties to even."""
    codes = torch.round(values / step)
    if not torch.all(torch.isfinite(codes) & (codes >= -(2**31)) & (codes < 2**31)):
        raise ValueError("Quantized bias does not fit INT32")
    return codes.to(torch.int32)


def shift():
    i8 = ir.IntegerType.get_signless(8)
    zero = ir.DenseElementsAttr.get_splat(
        ir.RankedTensorType.get([1], i8), ir.IntegerAttr.get(i8, 0)
    )
    return tosa.ConstOp(zero).results[0]


def fp32_to_int8_quant_reference(node, symbol_table):
    values = symbol_table[(str(node.args[0]), 0)]
    multiplier = symbol_table[(str(node.args[1]), 0)]
    shape = list(node.tensor_meta["shape"])
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
    f32 = ir.RankedTensorType.get(list(node.tensor_meta["shape"]), ir.F32Type.get())
    values = tosa.CastOp(f32, codes).result
    return tosa.MulOp(f32, values, step, shift())


def int32_to_fp32_dequant_reference(node, symbol_table):
    codes = symbol_table[(str(node.args[0]), 0)]
    step = symbol_table[(str(node.args[1]), 0)]
    f32 = ir.RankedTensorType.get(list(node.tensor_meta["shape"]), ir.F32Type.get())
    values = tosa.CastOp(f32, codes).result
    return tosa.MulOp(f32, values, step, shift())


def fp32_to_int8_quant_op(node, symbol_table):
    i8 = ir.IntegerType.get_signless(8)
    return ir.Operation.create(
        "frontend.fp32_to_int8",
        results=[ir.RankedTensorType.get(list(node.tensor_meta["shape"]), i8)],
        operands=[symbol_table[(str(argument), 0)] for argument in node.args],
    ).result


def int8_to_fp32_dequant_op(node, symbol_table):
    f32 = ir.F32Type.get()
    return ir.Operation.create(
        "frontend.int8_to_fp32",
        results=[ir.RankedTensorType.get(list(node.tensor_meta["shape"]), f32)],
        operands=[symbol_table[(str(argument), 0)] for argument in node.args],
    ).result


def int32_to_fp32_dequant_op(node, symbol_table):
    f32 = ir.F32Type.get()
    return ir.Operation.create(
        "frontend.int32_to_fp32",
        results=[ir.RankedTensorType.get(list(node.tensor_meta["shape"]), f32)],
        operands=[symbol_table[(str(argument), 0)] for argument in node.args],
    ).result


ops_registry = {
    "FP32ToInt8QuantOp": fp32_to_int8_quant_op,
    "Int8ToFP32DequantOp": int8_to_fp32_dequant_op,
    "Int32ToFP32DequantOp": int32_to_fp32_dequant_op,
}

ref_registry = {
    "FP32ToInt8QuantOp": fp32_to_int8_quant_reference,
    "Int8ToFP32DequantOp": int8_to_fp32_dequant_reference,
    "Int32ToFP32DequantOp": int32_to_fp32_dequant_reference,
}

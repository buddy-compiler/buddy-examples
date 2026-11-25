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
from buddy.compiler.graph.type import TensorDType


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


class Int32ToInt8QuantOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType


class FP32ToInt32QuantOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType


class ScaledAddI8Op(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType


class LookupI8Op(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType


class LookupI32Op(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType


class SoftmaxI8Op(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ReduceType


class RowMaxU32Op(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ReduceType


class GatherRowsOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ReshapeType


class ConcatColumnsOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ConcatType


class FixedRequantOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType


class FixedUnaryOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType


class FixedBinaryOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType


class FixedMeanOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ReduceType


class FP32ToMXINT8QuantOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType


class MXINT8ToFP32DequantOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType


class MXINT5ToFP32DequantOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType


class MXINT8ActivationOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType


def mxint8_activation_op(node, symbol_table):
    return ir.Operation.create(
        "frontend.mxint8_activation",
        results=[
            ir.RankedTensorType.get(
                mlir_shape(node.tensor_meta["shape"]), ir.IntegerType.get_signless(8)
            )
        ],
        operands=[symbol_table[(name, 0)] for name in node.args[:3]],
        attributes={
            "axis": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), node.args[3]),
            "gated": ir.BoolAttr.get(node.args[4]),
        },
    ).result


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


def dequantize_mxint5(packed, axis):
    """Decode native signed INT5 blocks without intermediate INT8 rounding."""
    p = packed.movedim(axis, -1).contiguous().view(torch.uint8)
    if p.shape[-1] % 22:
        raise ValueError("INT5 weights require complete 32-value blocks")
    blocks = p.shape[-1] // 22
    words = p[..., : blocks * 20].long().reshape(*p.shape[:-1], blocks, 4, 5)
    word = sum(words[..., i] << (8 * i) for i in range(5))
    codes = torch.stack([((word >> (5 * i)) & 31) for i in range(8)], -1)
    codes = ((codes ^ 16) - 16).reshape(*p.shape[:-1], blocks, 32)
    factors = p[..., blocks * 20 : blocks * 21].long()[..., None]
    exponents = p[..., blocks * 21 :].long()[..., None]
    if (
        torch.any(codes == -16)
        or torch.any((factors < 1) | (factors > 137))
        or torch.any(exponents == 255)
    ):
        raise ValueError("Invalid native INT5 weight encoding")
    value = torch.ldexp((codes * factors).double(), exponents - 131)
    return value.flatten(-2).float().movedim(-1, axis)


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


def fp32_to_int32_quant_op(node, symbol_table):
    return ir.Operation.create(
        "frontend.fp32_to_int32",
        results=[
            ir.RankedTensorType.get(
                mlir_shape(node.tensor_meta["shape"]), ir.IntegerType.get_signless(32)
            )
        ],
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


def mxint5_to_fp32_dequant_op(node, symbol_table):
    return ir.Operation.create(
        "frontend.mxint5_to_fp32",
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


def int32_to_int8_quant_op(node, symbol_table):
    return ir.Operation.create(
        "frontend.int32_to_int8",
        results=[
            ir.RankedTensorType.get(
                mlir_shape(node.tensor_meta["shape"]), ir.IntegerType.get_signless(8)
            )
        ],
        operands=[symbol_table[(str(argument), 0)] for argument in node.args],
    ).result


def scaled_add_i8_op(node, symbol_table):
    return ir.Operation.create(
        "frontend.scaled_add_i8",
        results=[
            ir.RankedTensorType.get(
                mlir_shape(node.tensor_meta["shape"]), ir.IntegerType.get_signless(8)
            )
        ],
        operands=[symbol_table[(str(argument), 0)] for argument in node.args[:2]],
        attributes={
            "lhs_ratio": ir.FloatAttr.get(ir.F32Type.get(), node.args[2]),
            "rhs_ratio": ir.FloatAttr.get(ir.F32Type.get(), node.args[3]),
        },
    ).result


def lookup_i8_op(node, symbol_table):
    return ir.Operation.create(
        "frontend.lookup_i8",
        results=[
            ir.RankedTensorType.get(
                mlir_shape(node.tensor_meta["shape"]), ir.IntegerType.get_signless(8)
            )
        ],
        operands=[symbol_table[(str(argument), 0)] for argument in node.args],
    ).result


def nonlinear_op(node, symbol_table):
    softmax = isinstance(node, SoftmaxI8Op)
    return ir.Operation.create(
        (
            "frontend.row_max_u32"
            if isinstance(node, RowMaxU32Op)
            else "frontend.softmax_i8" if softmax else "frontend.lookup_i32"
        ),
        results=[
            ir.RankedTensorType.get(
                mlir_shape(node.tensor_meta["shape"]),
                ir.IntegerType.get_signless(8 if softmax else 32),
            )
        ],
        operands=[symbol_table[(str(argument), 0)] for argument in node.args[:2]],
        attributes=(
            {
                "multiplier": ir.IntegerAttr.get(
                    ir.IntegerType.get_signless(64), node.args[2]
                )
            }
            if softmax
            else {}
        ),
    ).result


def gather_rows_op(node, symbol_table):
    element = {
        TensorDType.Float32: ir.F32Type.get(),
        TensorDType.Int32: ir.IntegerType.get_signless(32),
        TensorDType.Int8: ir.IntegerType.get_signless(8),
    }[node.tensor_meta["dtype"]]
    return ir.Operation.create(
        (
            "frontend.concat_columns"
            if isinstance(node, ConcatColumnsOp)
            else "frontend.gather_rows"
        ),
        results=[
            ir.RankedTensorType.get(mlir_shape(node.tensor_meta["shape"]), element)
        ],
        operands=[symbol_table[(str(argument), 0)] for argument in node.args],
    ).result


def fixed_op(node, symbol_table):
    i64 = ir.IntegerType.get_signless(64)
    attributes = {}
    if isinstance(node, FixedRequantOp):
        name, count = "fixed_requant", 3
        attributes["fraction"] = ir.IntegerAttr.get(i64, node.args[3])
    elif isinstance(node, FixedUnaryOp):
        name, count = "fixed_unary", 1
        attributes["kind"] = ir.StringAttr.get(node.args[1])
        attributes["fraction"] = ir.IntegerAttr.get(i64, node.args[2])
    elif isinstance(node, FixedBinaryOp):
        name, count = "fixed_binary", 2
        attributes["kind"] = ir.StringAttr.get(node.args[2])
        attributes["fraction"] = ir.IntegerAttr.get(i64, node.args[3])
    elif isinstance(node, FixedMeanOp):
        name, count = "fixed_mean", 1
    else:
        raise TypeError(type(node))
    return ir.Operation.create(
        "frontend." + name,
        results=[
            ir.RankedTensorType.get(
                mlir_shape(node.tensor_meta["shape"]), ir.IntegerType.get_signless(32)
            )
        ],
        operands=[symbol_table[(str(argument), 0)] for argument in node.args[:count]],
        attributes=attributes,
    ).result


ops_registry = {
    "MXINT8ActivationOp": mxint8_activation_op,
    "ScaledAddI8Op": scaled_add_i8_op,
    "FP32ToInt32QuantOp": fp32_to_int32_quant_op,
    "LookupI32Op": nonlinear_op,
    "SoftmaxI8Op": nonlinear_op,
    "RowMaxU32Op": nonlinear_op,
    "GatherRowsOp": gather_rows_op,
    "ConcatColumnsOp": gather_rows_op,
    "LookupI8Op": lookup_i8_op,
    "Int32ToInt8QuantOp": int32_to_int8_quant_op,
    "FixedRequantOp": fixed_op,
    "FixedUnaryOp": fixed_op,
    "FixedBinaryOp": fixed_op,
    "FixedMeanOp": fixed_op,
    "FP32ToInt8QuantOp": fp32_to_int8_quant_op,
    "Int8ToFP32DequantOp": int8_to_fp32_dequant_op,
    "Int32ToFP32DequantOp": int32_to_fp32_dequant_op,
    "FP32ToMXINT8QuantOp": fp32_to_mxint8_quant_op,
    "MXINT8ToFP32DequantOp": mxint8_to_fp32_dequant_op,
    "MXINT5ToFP32DequantOp": mxint5_to_fp32_dequant_op,
}

ref_registry = {
    "FP32ToInt8QuantOp": fp32_to_int8_quant_reference,
    "Int8ToFP32DequantOp": int8_to_fp32_dequant_reference,
    "Int32ToFP32DequantOp": int32_to_fp32_dequant_reference,
}

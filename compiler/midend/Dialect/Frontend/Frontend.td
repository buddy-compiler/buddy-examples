//===- Frontend.td - Frontend fake operations --------------*- tablegen -*-===//
//
// The operations a frontend inserts into a graph that no MLIR dialect has:
// the quantization of compiler/frontend/quant.py and the trace points of the
// Buddy importer (buddy.compiler.trace). They are fake: no hardware runs them
// as they are. A design recognizes the ones its hardware runs in its own
// convert-linalg-to-{device,hart,acc}; lower-frontend-fake-op, run after the
// design's passes, computes every remaining one in software.
//
//===----------------------------------------------------------------------===//

#ifndef FRONTEND_OPS
#define FRONTEND_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Frontend_Dialect : Dialect {
  let name = "frontend";
  let cppNamespace = "::buddy::frontend";
  let summary = "Quantization and trace points a frontend inserts";
  // Keep the inherent attributes in the attribute dictionary of the
  // assembly, as the frontend writes them.
  let useStrictPropertiesInAssemblyFormat = 0;
}

class Frontend_Op<string mnemonic, list<Trait> traits = []>
    : Op<Frontend_Dialect, mnemonic, traits>;

def FixedNormMXINT8Op : Frontend_Op<"fixed_norm_mxint8", [Pure]> {
  let summary = "Q16 integer LayerNorm with packed MXINT8 output";
  let description = [{
    The last input axis is normalized with RNE Q16 input and affine values,
    integer mean and variance, and an integer square root. Coefficients are
    little-endian bytes: positive u64 epsilon in Q32, eight reserved zero
    bytes, K signed Q16 gamma words, then K signed Q16 beta words.
    Each output row contains K INT8 codes followed by K/32 E8M0 exponents.
  }];
  let arguments = (ins RankedTensorOf<[F32]>:$input,
                       RankedTensorOf<[I8]>:$coefficients);
  let results = (outs RankedTensorOf<[I8]>:$result);
}

//===----------------------------------------------------------------------===//
// Symmetric INT8 quantization. A scale broadcasts against the values along
// its dimensions of size one. The quantization operations print in the
// generic form, as the frontend writes them.
//===----------------------------------------------------------------------===//

def MXINT8ActivationOp : Frontend_Op<"mxint8_activation", [Pure]> {
  let summary = "Integer table activation and optional gate on packed MXINT8";
  let arguments = (ins RankedTensorOf<[I8]>:$input, RankedTensorOf<[I8]>:$gate, RankedTensorOf<[I32]>:$table,
      I64Attr:$axis, BoolAttr:$gated);
  let results = (outs RankedTensorOf<[I8]>:$result);
}

def FP32ToInt8Op : Frontend_Op<"fp32_to_int8", [Pure]> {
  let summary = "Quantizes FP32 values to INT8";
  let description = [{
    `fp32_to_int8(x, multiplier) = clamp(round_half_even(x * multiplier),
    -128, 127)`, FP32ToInt8QuantOp of compiler/frontend/quant.py.
  }];
  let arguments = (ins RankedTensorOf<[F32]>:$input,
                       RankedTensorOf<[F32]>:$multiplier);
  let results = (outs RankedTensorOf<[I8]>:$result);
}

def Int32ToInt8Op : Frontend_Op<"int32_to_int8", [Pure]> {
  let summary = "Quantizes INT32 values with FP32 multipliers to INT8";
  let arguments = (ins RankedTensorOf<[I32]>:$input, RankedTensorOf<[F32]>:$multiplier);
  let results = (outs RankedTensorOf<[I8]>:$result);
}

def FP32ToInt32Op : Frontend_Op<"fp32_to_int32", [Pure]> {
  let summary = "FP32 scaling followed by ties-to-even rounding and INT32 saturation";
  let arguments = (ins RankedTensorOf<[F32]>:$input, RankedTensorOf<[F32]>:$multiplier);
  let results = (outs RankedTensorOf<[I32]>:$result);
}

def ScaledAddI8Op : Frontend_Op<"scaled_add_i8", [Pure]> {
  let summary = "INT8 sum with positive FP32 ratios, ties-to-even rounding and saturation";
  let arguments = (ins RankedTensorOf<[I8]>:$lhs, RankedTensorOf<[I8]>:$rhs,
                       F32Attr:$lhs_ratio, F32Attr:$rhs_ratio);
  let results = (outs RankedTensorOf<[I8]>:$result);
}

def LookupI8Op : Frontend_Op<"lookup_i8", [Pure]> {
  let summary = "Map bytes through shared or per-channel 256-entry INT8 tables";
  let arguments = (ins RankedTensorOf<[I8]>:$input, RankedTensorOf<[I8]>:$table);
  let results = (outs RankedTensorOf<[I8]>:$result);
}

def LookupI32Op : Frontend_Op<"lookup_i32", [Pure]> {
  let summary = "Clamped signed ten-bit indices into a 1024-entry INT32 table";
  let arguments = (ins RankedTensorOf<[I32]>:$input, RankedTensorOf<[I32]>:$table);
  let results = (outs RankedTensorOf<[I32]>:$result);
}

def SoftmaxI8Op : Frontend_Op<"softmax_i8", [Pure]> {
  let summary = "Max-relative Q30 table lookup and integer normalization to INT8";
  let arguments = (ins RankedTensorOf<[I32]>:$input, RankedTensorOf<[I32]>:$table,
                       I64Attr:$multiplier);
  let results = (outs RankedTensorOf<[I8]>:$result);
}

def IntegerSoftmaxOp : Frontend_Op<"integer_softmax", [Pure]> {
  let summary = "Q8 score lookup and Q15 probability normalization on at most 128 columns";
  let description = [{
    Clamp input scores to [-4096, 4096] and round to Q8 with ties to even.
    Subtract the integer row maximum, negate, and clamp indices to [0, 4095].
    The table contains round(exp(-i/256)*32768) as 4096 signed I32 words.
    Normalize each weight w by the row sum s using (w*32768+s/2)/s with
    integer division, then return the exact Q15 value in F32 representation.
    Each row has between one and 128 scores, including a dynamic row width
    whose upper bound the design lowering establishes.
    No floating point arithmetic is required during execution.
  }];
  let arguments = (ins RankedTensorOf<[F32]>:$input,
                       RankedTensorOf<[I32]>:$table);
  let results = (outs RankedTensorOf<[F32]>:$result);
  let hasVerifier = 1;
}

def RowMaxU32Op : Frontend_Op<"row_max_u32", [Pure]> {
  let summary = "Unsigned maximum of each nonempty INT32 matrix row";
  let arguments = (ins RankedTensorOf<[I32]>:$input);
  let results = (outs RankedTensorOf<[I32]>:$result);
}

def GatherRowsOp : Frontend_Op<"gather_rows", [Pure]> {
  let summary = "Gather valid row indices from logically concatenated matrices with contiguous rows";
  let arguments = (ins RankedTensorOf<[I64]>:$indices,
                       Variadic<RankedTensorOf<[I8, I32, F32]>>:$inputs);
  let results = (outs RankedTensorOf<[I8, I32, F32]>:$result);
}

def ConcatColumnsOp : Frontend_Op<"concat_columns", [Pure]> {
  let summary = "Concatenate matrix columns while preserving row order";
  let arguments = (ins Variadic<RankedTensorOf<[I8, I32, F32]>>:$inputs);
  let results = (outs RankedTensorOf<[I8, I32, F32]>:$result);
}

def FixedRequantOp : Frontend_Op<"fixed_requant", [Pure]> {
  let summary = "Per-channel or repeating sixteen-channel integer scaling, RNE and INT32 saturation";
  let arguments = (ins RankedTensorOf<[I32]>:$input,
                       RankedTensorOf<[I32]>:$multipliers, RankedTensorOf<[I32]>:$shifts,
                       I64Attr:$fraction);
  let results = (outs RankedTensorOf<[I32]>:$result);
}

def FixedUnaryOp : Frontend_Op<"fixed_unary", [Pure]> {
  let summary = "Integer fixed-point hardswish, hardsigmoid or relu";
  let arguments = (ins RankedTensorOf<[I32]>:$input, StrAttr:$kind, I64Attr:$fraction);
  let results = (outs RankedTensorOf<[I32]>:$result);
}

def FixedBinaryOp : Frontend_Op<"fixed_binary", [Pure]> {
  let summary = "Saturating fixed-point add or RNE multiply";
  let arguments = (ins RankedTensorOf<[I32]>:$lhs, RankedTensorOf<[I32]>:$rhs,
                       StrAttr:$kind, I64Attr:$fraction);
  let results = (outs RankedTensorOf<[I32]>:$result);
}

def FixedMeanOp : Frontend_Op<"fixed_mean", [Pure]> {
  let summary = "NHWC spatial mean with INT64 accumulation and RNE";
  let arguments = (ins RankedTensorOf<[I32]>:$input);
  let results = (outs RankedTensorOf<[I32]>:$result);
}

def Int8ToFP32Op : Frontend_Op<"int8_to_fp32", [Pure]> {
  let summary = "Dequantizes INT8 codes to FP32";
  let description = [{
    `int8_to_fp32(q, step) = q * step`, Int8ToFP32DequantOp of
    compiler/frontend/quant.py.
  }];
  let arguments = (ins RankedTensorOf<[I8]>:$input,
                       RankedTensorOf<[F32]>:$step);
  let results = (outs RankedTensorOf<[F32]>:$result);
}

def Int32ToFP32Op : Frontend_Op<"int32_to_fp32", [Pure]> {
  let summary = "Dequantizes INT32 accumulators to FP32";
  let description = [{
    `int32_to_fp32(q, step) = q * step`, Int32ToFP32DequantOp of
    compiler/frontend/quant.py: an INT32 accumulation of INT8 products, step
    the product of the steps of its two operands.
  }];
  let arguments = (ins RankedTensorOf<[I32]>:$input,
                       RankedTensorOf<[F32]>:$step);
  let results = (outs RankedTensorOf<[F32]>:$result);
}

//===----------------------------------------------------------------------===//
// MXINT8 (OCP MX, INT8 elements). Along `axis`, every 32 consecutive values
// share one exponent e = floor(log2(max |x|)) - 6, at least -127, and 0 for
// an all-zero block; each value is the code
// clamp(round_half_even(x * 2^-e), -127, 127), and code * 2^e is its value.
// The packed INT8 tensor holds along `axis` the K codes, then the K / 32
// bytes e + 127. A K that is no multiple of 32 (a dynamic token count) is
// zero-padded to Kp = ceil(K / 32) * 32: a zero changes neither the maximum of
// its block nor any product, so the packed axis holds Kp codes and Kp / 32
// exponents and unpacks to the Kp values, zeros after K.
//===----------------------------------------------------------------------===//

def FP32ToMXINT8Op : Frontend_Op<"fp32_to_mxint8", [Pure,
    DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface,
                              ["reifyResultShapes"]>]> {
  let summary = "Packs FP32 values to MXINT8 along an axis";
  let description = [{
    FP32ToMXINT8QuantOp of compiler/frontend/quant.py (quantize_mxint8 is its
    reference on offline values).
  }];
  let arguments = (ins RankedTensorOf<[F32]>:$input, I64Attr:$axis);
  let results = (outs RankedTensorOf<[I8]>:$result);
}

def MXINT8ToFP32Op : Frontend_Op<"mxint8_to_fp32", [Pure,
    DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface,
                              ["reifyResultShapes"]>]> {
  let summary = "Unpacks MXINT8 to FP32 values along an axis";
  let description = [{
    MXINT8ToFP32DequantOp of compiler/frontend/quant.py (dequantize_mxint8 is
    its reference on offline values).
  }];
  let arguments = (ins RankedTensorOf<[I8]>:$input, I64Attr:$axis);
  let results = (outs RankedTensorOf<[F32]>:$result);
}

def MXFP4ToFP32Op : Frontend_Op<"mxfp4_to_fp32", [Pure,
    DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface,
                              ["reifyResultShapes"]>]> {
  let summary = "Decodes MXFP4: E2M1 codes and E8M0 scales";
  let description = [{
    Each 32-value block has 16 code bytes, two E2M1 codes a byte (the even
    value in the low nibble), and an E8M0 exponent. Along axis, all code bytes
    precede all exponents: K values take K / 2 + K / 32 bytes. A value is the
    E2M1 value of its code times 2^(exponent - 127). compiler/frontend/mxfp4.py.
  }];
  let arguments = (ins RankedTensorOf<[I8]>:$input, I64Attr:$axis);
  let results = (outs RankedTensorOf<[F32]>:$result);
}

def MXINT5ToFP32Op : Frontend_Op<"mxint5_to_fp32", [Pure,
    DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface,
                              ["reifyResultShapes"]>]> {
  let summary = "Decodes signed INT5 codes, integer factors and E8M0 scales";
  let description = [{
    Each 32-value block has 20 code bytes, an unsigned factor and an E8M0
    exponent. Along axis, all codes precede all factors and all exponents.
    A value is code * factor * 2^(exponent - 131).
  }];
  let arguments = (ins RankedTensorOf<[I8]>:$input, I64Attr:$axis);
  let results = (outs RankedTensorOf<[F32]>:$result);
}

//===----------------------------------------------------------------------===//
// Trace points: a trace_start before a traced operation, a trace_end after
// it. `id_path` names the trace (its files, at most four levels); `cycle`
// records the cycles from start to end, `tensor` writes the value at the end
// (f32, bf16, i8 or i32 of a static shape). The runtime of stack/runtime/trace
// implements both.
//===----------------------------------------------------------------------===//

def TraceStartOp : Frontend_Op<"trace_start", [MemoryEffects<[MemWrite]>]> {
  let summary = "Starts a traced region";
  let arguments = (ins I64Attr:$id, I64ArrayAttr:$id_path, BoolAttr:$cycle,
                       BoolAttr:$tensor, OptionalAttr<StrAttr>:$tag);
  let assemblyFormat = "attr-dict";
}

def TraceEndOp : Frontend_Op<"trace_end", [SameOperandsAndResultType,
    MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Ends a traced region and forwards the traced value";
  let arguments = (ins AnyType:$input, I64Attr:$id, I64ArrayAttr:$id_path,
                       BoolAttr:$cycle, BoolAttr:$tensor,
                       OptionalAttr<StrAttr>:$tag);
  let results = (outs AnyType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif // FRONTEND_OPS

//===- Frontend.td - Frontend fake operations --------------*- tablegen -*-===//
//
// The operations a frontend inserts into a graph that no MLIR dialect has:
// the quantization of compiler/frontend/quant.py and the trace points of the
// Buddy importer (buddy.compiler.trace). They are fake: no hardware runs them
// as they are. A design recognizes the ones its hardware runs in its own
// convert-linalg-to-{device,core,acc}; lower-frontend-fake-op, run after the
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

//===----------------------------------------------------------------------===//
// Symmetric INT8 quantization. A scale broadcasts against the values along
// its dimensions of size one. The quantization operations print in the
// generic form, as the frontend writes them.
//===----------------------------------------------------------------------===//

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

def FP32ToMXINT8Op : Frontend_Op<"fp32_to_mxint8", [Pure]> {
  let summary = "Packs FP32 values to MXINT8 along an axis";
  let description = [{
    FP32ToMXINT8QuantOp of compiler/frontend/quant.py (quantize_mxint8 is its
    reference on offline values).
  }];
  let arguments = (ins RankedTensorOf<[F32]>:$input, I64Attr:$axis);
  let results = (outs RankedTensorOf<[I8]>:$result);
}

def MXINT8ToFP32Op : Frontend_Op<"mxint8_to_fp32", [Pure]> {
  let summary = "Unpacks MXINT8 to FP32 values along an axis";
  let description = [{
    MXINT8ToFP32DequantOp of compiler/frontend/quant.py (dequantize_mxint8 is
    its reference on offline values).
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

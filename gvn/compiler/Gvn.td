//===- Gvn.td - GVN dialect ---------------------------------*- tablegen -*-===//
//
// The marks of the stimulus path: a check names a value a slice of the
// workload starts or ends at.
//
//===----------------------------------------------------------------------===//

#ifndef GVN_OPS
#define GVN_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Gvn_Dialect : Dialect {
  let name = "gvn";
  let cppNamespace = "::gvn";
  let summary = "Slice boundaries of the stimulus path";
}

def CheckOp : Op<Gvn_Dialect, "check", [MemoryEffects<[MemWrite]>]> {
  let summary = "A value a slice starts or ends at";
  let description = [{
    `id` is the line and column of the linalg operation whose value `value`
    equals, in the linalg input of the hart compiler.
  }];
  let arguments = (ins AnyRankedTensor:$value, StrAttr:$id);
  let assemblyFormat = "$value `id` $id attr-dict `:` type($value)";
}

#endif // GVN_OPS

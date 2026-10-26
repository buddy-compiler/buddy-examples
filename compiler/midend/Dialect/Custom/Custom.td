//===- Custom.td - Computation-level custom operations ------*- tablegen -*-===//

#ifndef CUSTOM_OPS
#define CUSTOM_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Custom_Dialect : Dialect {
  let name = "custom";
  let cppNamespace = "::buddy::custom";
  let summary = "Tensor computations selected by a design's form-custom-ops";
}

def CustomOp : Op<Custom_Dialect, "op", [Pure]> {
  let summary = "A computation whose kind is defined by the design";
  let arguments = (ins StrAttr:$kind, Variadic<AnyRankedTensor>:$inputs);
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
  let assemblyFormat = [{
    $kind `(` $inputs `)` prop-dict attr-dict `:`
      functional-type($inputs, $outputs)
  }];
}

#endif // CUSTOM_OPS

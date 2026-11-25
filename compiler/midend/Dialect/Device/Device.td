//===- Device.td - Graphs on the harts of a device ----------*- tablegen -*-===//
//
// The device level of the midend. A device program is a function of the
// control hart: it launches graphs on harts and moves values between harts;
// the plain operations between them run on the control hart itself. The
// frontend names the class of hart each graph needs; the design's
// convert-linalg-to-device places every launch on a hart of that class and
// inserts a transfer for every value a hart reads from another; lower-device
// lowers both to the job runtime and the device runtime of the design.
//
// Both operations are in destination style: a launch writes its results into
// its `outs`, a transfer its value into its destination on the receiving
// hart. Bufferized, they read and write buffers and have no results, so
// lower-device lowers them after bufferization, when every value is a buffer.
//
//===----------------------------------------------------------------------===//

#ifndef DEVICE_OPS
#define DEVICE_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"

def Device_Dialect : Dialect {
  let name = "device";
  let cppNamespace = "::buddy::device";
  let summary = "Graphs launched on the harts of a device and the values "
                "transferred between them";
}

def Device_LaunchOp : Op<Device_Dialect, "launch",
    [AttrSizedOperandSegments, DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Runs a graph on a hart";
  let description = [{
    Runs the function `callee` on one hart with `inputs`; its results are
    written into `outs`. Before placement the launch names the class of hart
    the graph needs (`hart_class`), after placement the index of the hart in
    the device (`hart`); a launch has exactly one of them. Bufferized, the
    launch has no results and the callee takes the buffers of `outs` after
    its inputs.

    ```mlir
    %t = device.launch @vision(%w, %image) outs(%e) <{hart_class = "dense"}>
        : (tensor<64xf32>, tensor<1x3x64x64xf32>, tensor<1x4x128xf32>)
        -> tensor<1x4x128xf32>
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$inputs,
                   Variadic<AnyType>:$outs, OptionalAttr<StrAttr>:$hart_class,
                   OptionalAttr<I64Attr>:$hart);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    $callee `(` $inputs `)` `outs` `(` $outs `)` prop-dict attr-dict `:`
      functional-type(operands, results)
  }];
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getOutsMutable();
    }
  }];
  let hasVerifier = 1;
}

def Device_TransferOp : Op<Device_Dialect, "transfer",
    [DestinationStyleOpInterface]> {
  let summary = "Moves a value from one hart to another";
  let description = [{
    Writes `source`, a value of hart `from`, into `target`, a value of hart
    `to`.

    ```mlir
    %k = device.transfer %v to(%e) <{from = 1 : i64, to = 3 : i64}>
        : (tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> tensor<1x8x64xf32>
    ```
  }];
  let arguments = (ins AnyType:$source, AnyType:$target, I64Attr:$from,
                   I64Attr:$to);
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let assemblyFormat = [{
    $source `to` `(` $target `)` prop-dict attr-dict `:`
      functional-type(operands, results)
  }];
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getTargetMutable();
    }
  }];
}

#endif // DEVICE_OPS

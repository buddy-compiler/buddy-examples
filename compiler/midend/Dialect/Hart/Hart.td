//===- Hart.td - Subgraphs on the hardware threads of a hart -*- tablegen -*-===//
//
// The hart level of the midend. A graph runs on the CPU of its hart; the
// design places the parts of it that the hardware threads of the hart can
// compute on those threads, as subgraphs of their own (the units the threads
// run), and schedules them: which thread runs which unit, in which order,
// which units run at the same time, and how values pass between the hart CPU
// and a thread and between two threads. A thread sees plain DDR: a value it
// reads or writes is a buffer in DDR.
//
// The order is that of the values. A launch starts once the values it reads
// are there and runs asynchronously: the hart CPU goes on, launches that read
// no value of each other may run at the same time on different threads, and
// the CPU reads a result only through a transfer, which waits for its launch.
// A value of a thread reaches another thread only through a transfer between
// the two; lower-hart lowers it to the logical transfer of the kernel, bound
// by the compiled program.
//
// Both operations are in destination style: a launch writes its results into
// its `outs`, a transfer its value into its destination. Bufferized, they
// read and write buffers and have no results, so lower-hart lowers them
// after bufferization, when every value is a buffer.
//
//===----------------------------------------------------------------------===//

#ifndef HART_OPS
#define HART_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Hart_Dialect : Dialect {
  let name = "hart";
  let cppNamespace = "::buddy::hart";
  let summary = "Subgraphs scheduled on the hardware threads of a hart and "
                "the values transferred between them and the hart CPU";
}

def Hart_LaunchOp : Op<Hart_Dialect, "launch",
    [AttrSizedOperandSegments, DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Runs a subgraph on a hardware thread of the hart";
  let description = [{
    Runs the function `callee` on the hardware thread `thread` of the hart
    (its index among the threads of the hart) with `inputs`, values of that
    thread (transferred to it, or results of its launches), then `outs`, the
    destinations it writes its results into and returns (destination passing:
    the hart CPU sizes them, dynamic sizes included). The launch is
    asynchronous: it starts once its inputs are there, and its results are
    there once it is done. Bufferized, the launch has no results and the
    callee takes the buffers of `inputs` and `outs`.

    ```mlir
    %y = hart.launch @graph_ht0(%x, %w) outs(%e) <{thread = 0 : i64}>
        : (tensor<16x64xf32>, tensor<64x66xi8>, tensor<16x64xf32>)
        -> tensor<16x64xf32>
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$inputs,
                   Variadic<AnyType>:$outs, I64Attr:$thread);
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

def Hart_TransferOp : Op<Hart_Dialect, "transfer",
    [DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Moves a value between the hart CPU and a hardware thread "
                "or between two hardware threads";
  let description = [{
    Writes `source` into `target`. `from` and `to` name hardware threads of
    the hart; the one left out is the hart CPU. A transfer from a thread
    waits for the launch that writes `source`; a transfer to a thread comes
    before the launches that read `target`.

    ```mlir
    // From the hart CPU to thread 0.
    %t = hart.transfer %x to(%e) <{to = 0 : i64}>
        : (tensor<16x64xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
    // From thread 0 to the hart CPU.
    %y = hart.transfer %r to(%f) <{from = 0 : i64}>
        : (tensor<16x64xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
    // From thread 0 to thread 1.
    %z = hart.transfer %r to(%g) <{from = 0 : i64, to = 1 : i64}>
        : (tensor<16x64xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
    ```
  }];
  let arguments = (ins AnyType:$source, AnyType:$target,
                   OptionalAttr<I64Attr>:$from, OptionalAttr<I64Attr>:$to);
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
  let hasVerifier = 1;
}

#endif // HART_OPS

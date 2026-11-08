//===- Acc.td - Acc dialect --------------------------------*- tablegen -*-===//
//
// The tensor computations a design selects, banks and data movement between
// memory and banks. The computations on banks are the operations of the Balls, which join the bank layer through
// the interfaces of BankInterfaces.td.
//
//===----------------------------------------------------------------------===//

#ifndef ACC_OPS
#define ACC_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// The namespace is bbacc: the OpenACC dialect of MLIR, which buddy-opt
// registers, owns acc.
def Acc_Dialect : Dialect {
  let name = "bbacc";
  let cppNamespace = "::buddy::acc";
  let summary = "Custom computations, banks and bank data movement";
}

class Acc_Op<string mnemonic, list<Trait> traits = []>
    : Op<Acc_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

//===----------------------------------------------------------------------===//
// Tensor computations: formed and lowered to banks by a design's
// convert-linalg-to-acc.
//===----------------------------------------------------------------------===//

def CustomOp : Op<Acc_Dialect, "custom", [Pure]> {
  let summary = "A computation whose kind is defined by the design";
  let arguments = (ins StrAttr:$kind, Variadic<AnyRankedTensor>:$inputs);
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
  let assemblyFormat = [{
    $kind `(` $inputs `)` prop-dict attr-dict `:`
      functional-type($inputs, $outputs)
  }];
}

//===----------------------------------------------------------------------===//
// Virtual banks: i64 SSA handles until assign-physical-banks.
// Bank 0 is the zero bank, like x0: it is never allocated, and an operation
// without a destination bank names bank 0.
//===----------------------------------------------------------------------===//

def BankAllocOp : Acc_Op<"bank_alloc"> {
  let summary = "Allocate a bank of row x col physical banks";
  let arguments = (ins I64Attr:$row, I64Attr:$col);
  let results = (outs I64:$bank);
  let assemblyFormat = "prop-dict attr-dict";
}

def BankReleaseOp : Acc_Op<"bank_release"> {
  let summary = "Release a bank";
  let arguments = (ins I64:$bank);
  let assemblyFormat = "$bank prop-dict attr-dict";
}

def BankMvinOp : Acc_Op<"bank_mvin"> {
  let summary = "Load `depth` rows from memory into a bank";
  let description = [{
    Row i of the bank reads 16 bytes at `input` + i * `stride` * 16 bytes.
  }];
  let arguments = (ins AnyMemRef:$input, I64:$bank, I64Attr:$depth,
                       I64Attr:$stride);
  let results = (outs I64:$bankOut);
  let assemblyFormat = "$input `,` $bank prop-dict attr-dict `:` type($input)";
}

def BankMvoutOp : Acc_Op<"bank_mvout"> {
  let summary = "Store `depth` rows of a bank to memory";
  let arguments = (ins AnyMemRef:$output, I64:$bank, I64Attr:$depth,
                       I64Attr:$stride);
  let results = (outs I64:$bankOut);
  let assemblyFormat = "$output `,` $bank prop-dict attr-dict `:` type($output)";
}

def BankTransferOp : Acc_Op<"bank_transfer"> {
  let summary = "Append the groups of a bank to another bank";
  let description = [{
    `source` is released; `target` holds its own groups followed by those of
    `source`. Gathers the operands of a computation into the banks it reads.
  }];
  let arguments = (ins I64:$source, I64:$target);
  let results = (outs I64:$targetOut);
  let assemblyFormat = "$source `,` $target attr-dict";
}

//===----------------------------------------------------------------------===//
// Physical banks: after assign-physical-banks, lowered by lower-acc.
//===----------------------------------------------------------------------===//

def MsetOp : Acc_Op<"mset"> {
  let summary = "Allocate or release a bank";
  let arguments = (ins I64Attr:$bank, BoolAttr:$alloc, I64Attr:$row,
                       I64Attr:$col);
  let assemblyFormat = "prop-dict attr-dict";
}

def MvinOp : Acc_Op<"mvin"> {
  let summary = "Load `depth` rows from memory into a bank";
  let arguments = (ins AnyMemRef:$input, I64Attr:$bank, I64Attr:$depth,
                       I64Attr:$stride);
  let assemblyFormat = "$input prop-dict attr-dict `:` type($input)";
}

def MvoutOp : Acc_Op<"mvout"> {
  let summary = "Store `depth` rows of a bank to memory";
  let arguments = (ins AnyMemRef:$output, I64Attr:$bank, I64Attr:$depth,
                       I64Attr:$stride);
  let assemblyFormat = "$output prop-dict attr-dict `:` type($output)";
}

def MsetTransferOp : Acc_Op<"mset_transfer"> {
  let summary = "Append the groups of bank `source` to bank `target`";
  let arguments = (ins I64Attr:$source, I64Attr:$target);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // ACC_OPS

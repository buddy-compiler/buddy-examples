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

def BankMvinGroupOp : Acc_Op<"bank_mvin_group"> {
  let summary = "Load `depth` rows from memory into group `group` of a bank";
  let description = [{
    Like bank_mvin, into the group `group` of a bank of several groups.
  }];
  let arguments = (ins AnyMemRef:$input, I64:$bank, I64Attr:$group,
                       I64Attr:$depth, I64Attr:$stride);
  let results = (outs I64:$bankOut);
  let assemblyFormat = "$input `,` $bank prop-dict attr-dict `:` type($input)";
}

def BankMvinStridedOp : Acc_Op<"bank_mvin_strided"> {
  let summary = "Strided byte rows between DDR and one bank group";
  let arguments = (ins AnyMemRef:$memory, I64:$bank,
                       Index:$first, Index:$depth, Index:$memory_stride,
                       I64Attr:$group, I64Attr:$bank_stride, Index:$bytes, Index:$block_log2);
  let results = (outs I64:$bankOut);
  let assemblyFormat = "$memory `,` $bank `at` $first `rows` $depth "
                       "`memory_stride` $memory_stride `bytes` $bytes `block_log2` $block_log2 prop-dict attr-dict `:` type($memory)";
}

def MvinStridedOp : Acc_Op<"mvin_strided"> {
  let summary = "Strided byte rows between DDR and one bank group";
  let arguments = (ins AnyMemRef:$memory, I64Attr:$bank,
                       Index:$first, Index:$depth, Index:$memory_stride,
                       I64Attr:$group, I64Attr:$bank_stride, Index:$bytes, Index:$block_log2);
  let assemblyFormat = "$memory `,` $bank `at` $first `rows` $depth "
                       "`memory_stride` $memory_stride `bytes` $bytes `block_log2` $block_log2 prop-dict attr-dict `:` type($memory)";
}

def BankMvoutStridedOp : Acc_Op<"bank_mvout_strided"> {
  let summary = "Strided byte rows between DDR and one bank group";
  let arguments = (ins AnyMemRef:$memory, I64:$bank,
                       Index:$first, Index:$depth, Index:$memory_stride,
                       I64Attr:$group, I64Attr:$bank_stride, Index:$bytes, Index:$block_log2);
  let results = (outs I64:$bankOut);
  let assemblyFormat = "$memory `,` $bank `at` $first `rows` $depth "
                       "`memory_stride` $memory_stride `bytes` $bytes `block_log2` $block_log2 prop-dict attr-dict `:` type($memory)";
}

def MvoutStridedOp : Acc_Op<"mvout_strided"> {
  let summary = "Strided byte rows between DDR and one bank group";
  let arguments = (ins AnyMemRef:$memory, I64Attr:$bank,
                       Index:$first, Index:$depth, Index:$memory_stride,
                       I64Attr:$group, I64Attr:$bank_stride, Index:$bytes, Index:$block_log2);
  let assemblyFormat = "$memory `,` $bank `at` $first `rows` $depth "
                       "`memory_stride` $memory_stride `bytes` $bytes `block_log2` $block_log2 prop-dict attr-dict `:` type($memory)";
}

def BankClearOp : Acc_Op<"bank_clear"> {
  let summary = "Zero the contents of an allocated bank";
  let arguments = (ins I64:$bank, I64Attr:$row, I64Attr:$col);
  let results = (outs I64:$bankOut);
  let assemblyFormat = "$bank prop-dict attr-dict";
}

def ClearOp : Acc_Op<"clear"> {
  let summary = "MSET clear on an allocated bank";
  let arguments = (ins I64Attr:$bank, I64Attr:$row, I64Attr:$col);
  let assemblyFormat = "prop-dict attr-dict";
}

def BankRunOperatorOp : Acc_Op<"bank_run_operator"> {
  let summary = "Run the operator program in bank `program` on the GP domain";
  let description = [{
    The operator reads the groups of `input` and writes the groups of `output`;
    the first group of `output` holds its launch descriptor (entry, code
    bytes, stack, eight arguments) when it starts.
  }];
  let arguments = (ins I64:$input, I64:$output, I64Attr:$program);
  let results = (outs I64:$outputOut);
  let assemblyFormat = "$input `,` $output prop-dict attr-dict";
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

def MvinGroupOp : Acc_Op<"mvin_group"> {
  let summary = "Load `depth` rows from memory into group `group` of a bank";
  let arguments = (ins AnyMemRef:$input, I64Attr:$bank, I64Attr:$group,
                       I64Attr:$depth, I64Attr:$stride);
  let assemblyFormat = "$input prop-dict attr-dict `:` type($input)";
}

def RunOperatorOp : Acc_Op<"run_operator"> {
  let summary = "Run the operator program in bank `program` on the GP domain";
  let arguments = (ins I64Attr:$input, I64Attr:$output, I64Attr:$program);
  let assemblyFormat = "prop-dict attr-dict";
}

//===----------------------------------------------------------------------===//
// Operator programs of the GP domain: an image (the 24-byte header, the code,
// the constants) moved into the program bank `program`, which is released by
// an mset. The program bank is outside the banks of assign-physical-banks.
//===----------------------------------------------------------------------===//

def MvinOperatorOp : Acc_Op<"mvin_operator"> {
  let summary = "Load `bytes` bytes of an operator image into bank `program`";
  let arguments = (ins AnyMemRef:$image, I64:$bytes, I64Attr:$program);
  let assemblyFormat = "$image `,` $bytes prop-dict attr-dict `:` type($image)";
}

#endif // ACC_OPS

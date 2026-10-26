//===- Buckyball.td - Buckyball dialect --------------------*- tablegen -*-===//
//
// Banks, data movement between memory and banks, and one opaque custom
// operation.
// A core's convert-linalg-to-buckyball maps custom tensor computations to
// bank operations. The Ball implementing a bank_custom kind lowers it in
// lower-buckyball.
//
//===----------------------------------------------------------------------===//

#ifndef BUCKYBALL_OPS
#define BUCKYBALL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Buckyball_Dialect : Dialect {
  let name = "buckyball";
  let cppNamespace = "::buddy::buckyball";
  let summary = "Banks, bank data movement and opaque custom bank operations";
}

class Buckyball_Op<string mnemonic, list<Trait> traits = []>
    : Op<Buckyball_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

//===----------------------------------------------------------------------===//
// Virtual banks: i64 SSA handles until assign-physical-banks.
//===----------------------------------------------------------------------===//

def BankAllocOp : Buckyball_Op<"bank_alloc"> {
  let summary = "Allocate a bank of row x col physical banks";
  let arguments = (ins I64Attr:$row, I64Attr:$col);
  let results = (outs I64:$bank);
  let assemblyFormat = "prop-dict attr-dict";
}

def BankReleaseOp : Buckyball_Op<"bank_release"> {
  let summary = "Release a bank";
  let arguments = (ins I64:$bank);
  let assemblyFormat = "$bank prop-dict attr-dict";
}

def BankMvinOp : Buckyball_Op<"bank_mvin"> {
  let summary = "Load `depth` rows from memory into a bank";
  let description = [{
    Row i of the bank reads 16 bytes at `input` + i * `stride` * 16 bytes.
  }];
  let arguments = (ins AnyMemRef:$input, I64:$bank, I64Attr:$depth,
                       I64Attr:$stride);
  let results = (outs I64:$bankOut);
  let assemblyFormat = "$input `,` $bank prop-dict attr-dict `:` type($input)";
}

def BankMvoutOp : Buckyball_Op<"bank_mvout"> {
  let summary = "Store `depth` rows of a bank to memory";
  let arguments = (ins AnyMemRef:$output, I64:$bank, I64Attr:$depth,
                       I64Attr:$stride);
  let results = (outs I64:$bankOut);
  let assemblyFormat = "$output `,` $bank prop-dict attr-dict `:` type($output)";
}

def BankCustomOp : Buckyball_Op<"bank_custom", [AttrSizedOperandSegments]> {
  let summary = "A computation of up to two source banks into one bank";
  let description = [{
    `kind` names the computation; the Ball implementing it lowers it. The
    other attributes and the scalar `arguments` belong to that computation.
    With `read_destination` the computation also reads the old destination,
    for instance to accumulate.
  }];
  let arguments = (ins Variadic<I64>:$sources, I64:$destination,
                       Variadic<AnyType>:$arguments, StrAttr:$kind,
                       UnitAttr:$read_destination);
  let results = (outs I64:$destinationOut);
  let assemblyFormat =
      "$kind `(` $sources `)` `->` $destination (`args` `(` $arguments^ `:` "
      "type($arguments) `)`)? prop-dict attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Physical banks: after assign-physical-banks, lowered by lower-buckyball.
//===----------------------------------------------------------------------===//

def MsetOp : Buckyball_Op<"mset"> {
  let summary = "Allocate or release a bank";
  let arguments = (ins I64Attr:$bank, BoolAttr:$alloc, I64Attr:$row,
                       I64Attr:$col);
  let assemblyFormat = "prop-dict attr-dict";
}

def MvinOp : Buckyball_Op<"mvin"> {
  let summary = "Load `depth` rows from memory into a bank";
  let arguments = (ins AnyMemRef:$input, I64Attr:$bank, I64Attr:$depth,
                       I64Attr:$stride);
  let assemblyFormat = "$input prop-dict attr-dict `:` type($input)";
}

def MvoutOp : Buckyball_Op<"mvout"> {
  let summary = "Store `depth` rows of a bank to memory";
  let arguments = (ins AnyMemRef:$output, I64Attr:$bank, I64Attr:$depth,
                       I64Attr:$stride);
  let assemblyFormat = "$output prop-dict attr-dict `:` type($output)";
}

def CustomOp : Buckyball_Op<"custom"> {
  let summary = "A computation of up to two source banks into one bank";
  let arguments = (ins Variadic<AnyType>:$arguments,
                       DenseI64ArrayAttr:$sources, I64Attr:$destination,
                       StrAttr:$kind, UnitAttr:$read_destination);
  let assemblyFormat =
      "$kind (`args` `(` $arguments^ `:` type($arguments) `)`)? prop-dict "
      "attr-dict";
}

#endif // BUCKYBALL_OPS

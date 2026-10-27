//===- BankInterfaces.td - How Ball operations join the bank layer -*- tablegen -*-===//
//
// A Ball is a backend of ISA-level operations. Each of its computations is a
// pair of operations of the Ball's own dialect:
//   - a virtual operation on bank SSA values, implementing BankSSALowering.
//     The bank passes see which banks it reads and writes, and
//     assign-physical-banks turns it into the physical operation;
//   - a physical operation on bank numbers, implementing
//     PhysicalBankAssignment. lower-buckyball turns it into the Ball's
//     instructions.
// A Ball reads at most two source banks; bank 0 is the zero bank.
//
//===----------------------------------------------------------------------===//

#ifndef BUCKYBALL_BANK_INTERFACES
#define BUCKYBALL_BANK_INTERFACES

include "mlir/IR/OpBase.td"

def BankSSALowering : OpInterface<"BankSSALowering"> {
  let cppNamespace = "::buddy::buckyball";
  let description = [{
    A Ball computation on virtual banks. Its one result is the destination
    bank after the computation.
  }];
  let methods = [
    InterfaceMethod<"The source banks it reads, at most two.",
                    "::llvm::SmallVector<::mlir::Value, 2>", "getSourceBanks">,
    InterfaceMethod<"The bank it writes.", "::mlir::Value",
                    "getDestinationBank">,
    InterfaceMethod<"Whether it also reads the old destination.", "bool",
                    "readsDestination">,
    InterfaceMethod<[{
        Creates the physical operation on the physical banks `sources` and
        `destination` at the insertion point of `builder`.
      }], "::mlir::Operation *", "buildPhysical",
      (ins "::mlir::OpBuilder &":$builder,
           "::llvm::ArrayRef<int64_t>":$sources, "int64_t":$destination)>,
  ];
  let verify = [{
    if ($_op->getNumResults() != 1 ||
        !$_op->getResult(0).getType().isInteger(64))
      return $_op->emitOpError("gives exactly the destination bank");
    if (::mlir::cast<ConcreteOp>($_op).getSourceBanks().size() > 2)
      return $_op->emitOpError("reads at most two source banks");
    return ::mlir::success();
  }];
}

def PhysicalBankAssignment : OpInterface<"PhysicalBankAssignment"> {
  let cppNamespace = "::buddy::buckyball";
  let description = [{
    A Ball computation on physical banks. An operation that touches no bank
    has no sources and destination 0, the zero bank.
  }];
  let methods = [
    InterfaceMethod<"The physical source banks it reads, at most two.",
                    "::llvm::SmallVector<int64_t, 2>", "getSourceBanks">,
    InterfaceMethod<"The physical bank it writes.", "int64_t",
                    "getDestinationBank">,
    InterfaceMethod<[{
        Emits the Ball instructions of the operation at the insertion point
        of `builder`, or reports why the Ball cannot run it.
      }], "::mlir::LogicalResult", "lowerToInstructions",
      (ins "::mlir::OpBuilder &":$builder)>,
  ];
  let verify = [{
    if (::mlir::cast<ConcreteOp>($_op).getSourceBanks().size() > 2)
      return $_op->emitOpError("reads at most two source banks");
    return ::mlir::success();
  }];
}

#endif // BUCKYBALL_BANK_INTERFACES

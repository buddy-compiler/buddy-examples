//===- PackedEntry.cpp - One entry for a program of any arity -------------===//
//
// gvn-packed-entry, on the LLVM dialect after -llvm-request-c-wrappers:
// adds for each program <p> (marked gvn.program) gvn_<p>(results,
// arguments), which calls _mlir_ciface_<p>(results, arguments[0], ...,
// arguments[n - 1]), so the host harness calls a program whatever the count
// of its arguments (the manifest of gvn-gpu-program).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
class PackedEntryPass
    : public PassWrapper<PackedEntryPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PackedEntryPass)
  StringRef getArgument() const final { return "gvn-packed-entry"; }
  StringRef getDescription() const final {
    return "Add gvn_<program>(results, arguments) calling each program";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<std::string> programs;
    for (auto function : module.getOps<LLVM::LLVMFuncOp>())
      // The C wrapper of a program carries its attributes too.
      if (function->hasAttr("gvn.program") &&
          !function.getSymName().starts_with("_mlir_ciface_"))
        programs.push_back(function.getSymName().str());
    if (programs.empty())
      llvm::report_fatal_error("gvn-packed-entry: no program");
    for (const std::string &name : programs) {
      auto program =
          module.lookupSymbol<LLVM::LLVMFuncOp>("_mlir_ciface_" + name);
      if (!program)
        llvm::report_fatal_error(
            llvm::Twine("gvn-packed-entry: no _mlir_ciface_") + name);
      Location loc = program.getLoc();
      OpBuilder builder(module.getContext());
      builder.setInsertionPointToEnd(module.getBody());
      auto pointer = LLVM::LLVMPointerType::get(module.getContext());
      auto entry = LLVM::LLVMFuncOp::create(
          builder, loc, "gvn_" + name,
          LLVM::LLVMFunctionType::get(
              LLVM::LLVMVoidType::get(module.getContext()),
              {pointer, pointer}));
      Block *body = entry.addEntryBlock(builder);
      builder.setInsertionPointToStart(body);
      SmallVector<Value> operands{body->getArgument(0)};
      for (unsigned argument = 1; argument < program.getNumArguments();
           ++argument) {
        Value address = LLVM::GEPOp::create(
            builder, loc, pointer, pointer, body->getArgument(1),
            ArrayRef<LLVM::GEPArg>{int32_t(argument - 1)});
        operands.push_back(
            LLVM::LoadOp::create(builder, loc, pointer, address));
      }
      LLVM::CallOp::create(builder, loc, program, operands);
      LLVM::ReturnOp::create(builder, loc, ValueRange{});
    }
  }
};
} // namespace

namespace gvn {
void registerPackedEntryPass() { PassRegistration<PackedEntryPass>(); }
} // namespace gvn

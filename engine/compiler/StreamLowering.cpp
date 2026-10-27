//===- StreamLowering.cpp - Ball instructions through the engine stream ---===//
//
// On a core with hardware threads, host code does not issue Ball instructions
// itself: each one becomes a stream_emit call, and the engine runs the batch on
// a hardware thread. Loaded into buddy-opt as a pass plugin.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Config/llvm-config.h"

using namespace mlir;

namespace {
// Ball instructions are the RISC-V intrinsics of the Buckyball extension.
int32_t funct7Of(LLVM::CallIntrinsicOp call) {
  StringRef name = call.getIntrin();
  if (name == "llvm.riscv.buckyball.custom") {
    auto constant = call.getArgs()[2].getDefiningOp<LLVM::ConstantOp>();
    return constant ? cast<IntegerAttr>(constant.getValue()).getInt() : -1;
  }
  return llvm::StringSwitch<int32_t>(name)
      .Case("llvm.riscv.bb.mvout", 16)
      .Case("llvm.riscv.bb.mset", 32)
      .Case("llvm.riscv.bb.mvin", 33)
      .Case("llvm.riscv.bb.mvin_mmio", 35)
      .Default(-1);
}

class StreamLowering
    : public PassWrapper<StreamLowering, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StreamLowering)
  StringRef getArgument() const final { return "lower-to-stream"; }
  StringRef getDescription() const final {
    return "Issue lowered Ball instructions through the engine instruction "
           "stream";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<LLVM::CallIntrinsicOp> instructions;
    module.walk([&](LLVM::CallIntrinsicOp call) {
      if (call.getIntrin().starts_with("llvm.riscv.bb.") ||
          call.getIntrin() == "llvm.riscv.buckyball.custom")
        instructions.push_back(call);
    });
    if (instructions.empty())
      return;
    OpBuilder builder(&getContext());
    auto type = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(&getContext()),
        {builder.getI32Type(), builder.getI64Type(), builder.getI64Type()});
    auto emit = module.lookupSymbol<LLVM::LLVMFuncOp>("stream_emit");
    if (emit && emit.getFunctionType() != type) {
      emit.emitError("stream_emit must have signature void(i32, i64, i64)");
      return signalPassFailure();
    }
    if (!emit) {
      builder.setInsertionPointToStart(module.getBody());
      emit = builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "stream_emit",
                                              type);
    }
    for (LLVM::CallIntrinsicOp call : instructions) {
      int32_t funct7 = funct7Of(call);
      if (funct7 <= 0 || funct7 > 127) {
        call.emitError("unsupported stream command encoding");
        return signalPassFailure();
      }
      builder.setInsertionPoint(call);
      Value code =
          builder.create<LLVM::ConstantOp>(call.getLoc(), builder.getI32Type(),
                                           builder.getI32IntegerAttr(funct7));
      builder.create<LLVM::CallOp>(
          call.getLoc(), emit,
          ValueRange{code, call.getArgs()[0], call.getArgs()[1]});
      // The funct7 immediate of a custom instruction is now in `code`.
      Operation *immediate = call.getArgs().size() == 3
                                 ? call.getArgs()[2].getDefiningOp()
                                 : nullptr;
      call.erase();
      if (immediate && immediate->use_empty())
        immediate->erase();
    }
  }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Engine", LLVM_VERSION_STRING,
          [] { PassRegistration<StreamLowering>(); }};
}

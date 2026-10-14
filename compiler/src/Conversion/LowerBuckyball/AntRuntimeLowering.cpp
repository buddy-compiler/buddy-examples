#include "Conversion/LowerBuckyball/LowerBuckyball.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace {
class AntRuntimeLowering
    : public PassWrapper<AntRuntimeLowering, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AntRuntimeLowering)
  StringRef getArgument() const final { return "lower-ant-runtime"; }
  StringRef getDescription() const final {
    return "Submit lowered NPU instructions from the controller to Ant";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> instructions;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef().starts_with("buckyball.intr."))
        instructions.push_back(op);
    });
    if (instructions.empty())
      return;
    OpBuilder builder(&getContext());
    auto type = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(&getContext()),
        {builder.getI32Type(), builder.getI64Type(), builder.getI64Type()});
    auto emit = module.lookupSymbol<LLVM::LLVMFuncOp>("ant_emit");
    if (emit && emit.getFunctionType() != type) {
      emit.emitError("ant_emit must have signature void(i32, i64, i64)");
      return signalPassFailure();
    }
    if (!emit) {
      builder.setInsertionPointToStart(module.getBody());
      emit =
          builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "ant_emit", type);
    }
    for (Operation *op : instructions) {
      if (op->getNumOperands() != 2 || op->getNumResults() != 0 ||
          !llvm::all_of(op->getOperandTypes(),
                        [](Type type) { return type.isInteger(64); })) {
        op->emitError("Ant submission requires two i64 operands and no result");
        return signalPassFailure();
      }
      StringRef name = op->getName().getStringRef().drop_front(15);
      int32_t funct7;
      if (name == "custom") {
        funct7 = cast<IntegerAttr>(op->getAttr("funct7")).getInt();
      } else {
        funct7 = llvm::StringSwitch<int32_t>(name)
                     .Case("mvout", 16)
                     .Case("mset", 32)
                     .Case("mvin", 33)
                     .Case("mvin_mmio", 35)
                     .Default(-1);
      }
      if (funct7 <= 0 || funct7 > 127) {
        op->emitError("unsupported Ant command encoding");
        return signalPassFailure();
      }
      builder.setInsertionPoint(op);
      Value code =
          builder.create<LLVM::ConstantOp>(op->getLoc(), builder.getI32Type(),
                                           builder.getI32IntegerAttr(funct7));
      builder.create<LLVM::CallOp>(
          op->getLoc(), emit,
          ValueRange{code, op->getOperand(0), op->getOperand(1)});
      op->erase();
    }
  }
};
} // namespace

void mlir::buddy::registerAntRuntimePass() {
  PassRegistration<AntRuntimeLowering>();
}

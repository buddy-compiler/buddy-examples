#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
class VerifyGPTargetPass
    : public PassWrapper<VerifyGPTargetPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyGPTargetPass)
  StringRef getArgument() const final { return "verify-gp-target"; }
  StringRef getDescription() const final {
    return "Check the types of a separately compiled GP image against its "
           "target";
  }
  void runOnOperation() override {
    const auto &config = target::getTarget();
    if (!config.rvvEnabled) {
      getOperation().emitError("GP image needs a GP domain on this target");
      return signalPassFailure();
    }
    if (config.fp64Enabled)
      return;
    bool fp64 = false;
    AttrTypeWalker types;
    types.addWalk([&](FloatType type) { fp64 |= type.isF64(); });
    WalkResult result = getOperation().walk([&](Operation *op) {
      types.walk(op->getAttrDictionary());
      for (Type type : op->getOperandTypes())
        types.walk(type);
      for (Type type : op->getResultTypes())
        types.walk(type);
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            types.walk(argument.getType());
      if (fp64) {
        op->emitError(
            "GP target disables FP64; f64 is not permitted in this image");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};
PassRegistration<VerifyGPTargetPass> verifyRegistration;

class PruneGPOperatorsPass
    : public PassWrapper<PruneGPOperatorsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PruneGPOperatorsPass)
  StringRef getArgument() const final { return "prune-gp-operators"; }
  StringRef getDescription() const final {
    return "Remove separately compiled GP code from the host module";
  }
  void runOnOperation() override {
    for (auto function :
         llvm::make_early_inc_range(getOperation().getOps<func::FuncOp>()))
      if (function->hasAttr("bbacc.gp_operator") ||
          function->hasAttr("bbacc.gp_entry"))
        function.erase();
    for (auto global :
         llvm::make_early_inc_range(getOperation().getOps<memref::GlobalOp>()))
      if (global->hasAttr("bbacc.gp_space"))
        global.erase();
  }
};
PassRegistration<PruneGPOperatorsPass> registration;
} // namespace

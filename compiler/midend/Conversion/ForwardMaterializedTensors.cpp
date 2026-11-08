#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
class ForwardMaterializedTensorsPass
    : public PassWrapper<ForwardMaterializedTensorsPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ForwardMaterializedTensorsPass)
  StringRef getArgument() const final { return "forward-materialized-tensors"; }
  StringRef getDescription() const final {
    return "Copy repeated outputs from the materialized tensor's current value";
  }
  void runOnOperation() override {
    DominanceInfo dominance(getOperation());
    getOperation().walk([&](bufferization::MaterializeInDestinationOp op) {
      if (op->getNumResults() != 1 ||
          op.getSource().getType() != op.getResult().getType())
        return;
      // CSE can make several graph outputs the same tensor. Once one is
      // materialized, subsequent copies read its updated SSA value, so they
      // do not require the destination's pre-write contents to stay live.
      for (OpOperand &use :
           llvm::make_early_inc_range(op.getSource().getUses())) {
        auto next =
            dyn_cast<bufferization::MaterializeInDestinationOp>(use.getOwner());
        if (next && &use == &next.getSourceOperand() &&
            dominance.properlyDominates(op.getOperation(), next.getOperation()))
          use.set(op.getResult());
      }
    });
  }
};
PassRegistration<ForwardMaterializedTensorsPass> registration;
} // namespace

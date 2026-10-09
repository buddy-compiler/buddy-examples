#include "Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <numeric>

using namespace mlir;

namespace {
class FusePointwisePass
    : public PassWrapper<FusePointwisePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusePointwisePass)

  StringRef getArgument() const final { return "fuse-pointwise"; }
  StringRef getDescription() const final {
    return "Fuse pointwise producers and sliced gathers without expanding work.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect, tosa::TosaDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    buddy::populateGatherPatterns(patterns);
    linalg::populateElementwiseOpsFusionPatterns(
        patterns, [](OpOperand *operand) {
          auto producer = operand->get().getDefiningOp<linalg::GenericOp>();
          auto consumer = dyn_cast<linalg::GenericOp>(operand->getOwner());
          if (!producer || !consumer || !producer->hasOneUse() ||
              producer.getNumParallelLoops() != producer.getNumLoops())
            return false;
          auto source = producer.getStaticLoopRanges();
          auto target = consumer.getStaticLoopRanges();
          if (llvm::any_of(source, ShapedType::isDynamic) ||
              llvm::any_of(target, ShapedType::isDynamic))
            return false;
          return std::accumulate(source.begin(), source.end(), int64_t(1),
                                 std::multiplies<int64_t>()) ==
                 std::accumulate(target.begin(), target.end(), int64_t(1),
                                 std::multiplies<int64_t>());
        });
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns),
            GreedyRewriteConfig().setUseTopDownTraversal())))
      signalPassFailure();
  }
};
} // namespace

void mlir::buddy::registerFusePointwisePass() {
  PassRegistration<FusePointwisePass>();
}

//===- FuseElementwise.cpp - Fuse one-to-one tensor computations ----------===//
//
// Keep reductions and broadcasts as materialization boundaries: fusing into
// them would repeat quantization or weight restoration for every output.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
class FuseElementwisePass
    : public PassWrapper<FuseElementwisePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseElementwisePass)

  StringRef getArgument() const final { return "fuse-elementwise"; }
  StringRef getDescription() const final {
    return "Fuse single-use elementwise chains without repeating computation";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    linalg::populateElementwiseOpsFusionPatterns(
        patterns, [](OpOperand *operand) {
          auto consumer = cast<linalg::GenericOp>(operand->getOwner());
          return operand->get().getDefiningOp()->hasOneUse() &&
                 consumer.getNumReductionLoops() == 0 &&
                 consumer.getMatchingIndexingMap(operand).isPermutation();
        });
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace mlir::x86 {
void registerFuseElementwisePass() { PassRegistration<FuseElementwisePass>(); }
} // namespace mlir::x86

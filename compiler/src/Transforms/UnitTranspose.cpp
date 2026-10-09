#include "Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
class FoldUnitTranspose : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics())
      return failure();
    auto input = cast<RankedTensorType>(op.getInput().getType());
    auto output = cast<RankedTensorType>(op->getResult(0).getType());
    if (!input.hasStaticShape())
      return failure();
    SmallVector<int64_t> before, after;
    for (int64_t axis = 0; axis < input.getRank(); ++axis)
      if (input.getDimSize(axis) != 1)
        before.push_back(axis);
    for (int64_t axis : op.getPermutation())
      if (input.getDimSize(axis) != 1)
        after.push_back(axis);
    if (before != after)
      return failure();
    if (input == output) {
      rewriter.replaceOp(op, op.getInput());
      return success();
    }
    ReassociationIndices axes;
    for (int64_t axis = 0; axis < input.getRank(); ++axis)
      axes.push_back(axis);
    SmallVector<ReassociationIndices> reassociation{axes};
    auto flatType =
        RankedTensorType::get({input.getNumElements()}, input.getElementType());
    Value flat = rewriter.create<tensor::CollapseShapeOp>(
        op.getLoc(), flatType, op.getInput(), reassociation);
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(op, output, flat,
                                                       reassociation);
    return success();
  }
};

class FoldUnitTransposePass
    : public PassWrapper<FoldUnitTransposePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldUnitTransposePass)
  StringRef getArgument() const final { return "fold-unit-transpose"; }
  StringRef getDescription() const final {
    return "Replace tensor transposes that only move unit dimensions with "
           "views.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldUnitTranspose>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::buddy::registerFoldUnitTransposePass() {
  PassRegistration<FoldUnitTransposePass>();
}

#include "Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
template <typename Op> struct Outline : OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const override {
    if (!getElementTypeOrSelf(op.getInputs()[0].getType()).isF32() ||
        !getElementTypeOrSelf(op.getInputs()[1].getType()).isF32() ||
        !getElementTypeOrSelf(op.getOutputs()[0].getType()).isF32())
      return failure();
    auto call = rewriter.create<linalg::GenericOp>(
        op.getLoc(), op.getResultTypes(), op.getInputs(), op.getOutputs(),
        op.getIndexingMapsArray(), op.getIteratorTypesArray(),
        [](OpBuilder &builder, Location loc, ValueRange args) {
          Value product = builder.create<arith::MulFOp>(loc, args[0], args[1]);
          Value sum = builder.create<arith::AddFOp>(loc, args[2], product);
          builder.create<linalg::YieldOp>(loc, sum);
        });
    call.setLibraryCallAttr(rewriter.getStringAttr("rvv_matmul"));
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

class OutlineRVVMatmulPass
    : public PassWrapper<OutlineRVVMatmulPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OutlineRVVMatmulPass)
  StringRef getArgument() const final { return "outline-rvv-matmul"; }
  StringRef getDescription() const final {
    return "Outline FP32 matmul with separate multiply/add into private RVV.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<Outline<linalg::MatmulOp>, Outline<linalg::BatchMatmulOp>>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::buddy::registerRVVMatmulPass() {
  PassRegistration<OutlineRVVMatmulPass>();
}

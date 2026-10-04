#include "Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace {
class FoldGatherSlice : public OpRewritePattern<tensor::ExtractSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp slice,
                                PatternRewriter &rewriter) const override {
    auto producer = slice.getSource().getDefiningOp<linalg::GenericOp>();
    if (!producer || !producer->hasOneUse() ||
        !producer.hasPureTensorSemantics() || producer.getNumDpsInputs() != 0 ||
        producer.getNumDpsInits() != 1 ||
        producer.getNumParallelLoops() != producer.getNumLoops() ||
        !producer.getIndexingMapsArray()[0].isIdentity() ||
        slice.getType().getRank() != slice.getSourceType().getRank() ||
        !slice.getType().hasStaticShape() ||
        llvm::any_of(slice.getStaticOffsets(), ShapedType::isDynamic) ||
        llvm::any_of(slice.getStaticStrides(), ShapedType::isDynamic))
      return failure();
    Block &body = producer.getRegion().front();
    if (!body.getArgument(0).use_empty() ||
        llvm::any_of(body.without_terminator(), [](Operation &operation) {
          return operation.getNumRegions() != 0 || !isPure(&operation);
        }))
      return failure();

    auto type = slice.getType();
    Value output = rewriter.create<tensor::EmptyOp>(
        slice.getLoc(), type.getShape(), type.getElementType(),
        type.getEncoding());
    auto gathered = cast<linalg::GenericOp>(
        mlir::clone(rewriter, producer, TypeRange{type}, ValueRange{output}));
    for (auto index : llvm::make_early_inc_range(
             gathered.getRegion().front().getOps<linalg::IndexOp>())) {
      int64_t offset = slice.getStaticOffsets()[index.getDim()];
      int64_t stride = slice.getStaticStrides()[index.getDim()];
      if (offset == 0 && stride == 1)
        continue;
      SmallVector<OpOperand *> uses;
      for (OpOperand &use : index.getResult().getUses())
        uses.push_back(&use);
      rewriter.setInsertionPointAfter(index);
      Value mapped = index.getResult();
      if (stride != 1)
        mapped = rewriter.create<arith::MulIOp>(
            slice.getLoc(), mapped,
            rewriter.create<arith::ConstantIndexOp>(slice.getLoc(), stride));
      if (offset != 0)
        mapped = rewriter.create<arith::AddIOp>(
            slice.getLoc(), mapped,
            rewriter.create<arith::ConstantIndexOp>(slice.getLoc(), offset));
      for (OpOperand *use : uses)
        use->set(mapped);
    }
    rewriter.replaceOp(slice, gathered->getResults());
    rewriter.eraseOp(producer);
    return success();
  }
};
class FoldGatherTranspose : public OpRewritePattern<tosa::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::TransposeOp transpose,
                                PatternRewriter &rewriter) const override {
    auto producer = transpose.getInput1().getDefiningOp<linalg::GenericOp>();
    auto type = dyn_cast<RankedTensorType>(transpose.getOutput().getType());
    if (!producer || !type || !producer->hasOneUse() ||
        !producer.hasPureTensorSemantics() || producer.getNumDpsInputs() != 0 ||
        producer.getNumDpsInits() != 1 ||
        producer.getNumParallelLoops() != producer.getNumLoops() ||
        !producer.getIndexingMapsArray()[0].isIdentity() ||
        !type.hasStaticShape() ||
        type.getRank() !=
            cast<RankedTensorType>(transpose.getInput1().getType()).getRank())
      return failure();
    Block &body = producer.getRegion().front();
    if (!body.getArgument(0).use_empty() ||
        llvm::any_of(body.without_terminator(), [](Operation &operation) {
          return operation.getNumRegions() != 0 || !isPure(&operation);
        }))
      return failure();

    SmallVector<int64_t> inverse(type.getRank());
    for (auto [dimension, source] : llvm::enumerate(transpose.getPerms()))
      inverse[source] = dimension;
    Value output = rewriter.create<tensor::EmptyOp>(
        transpose.getLoc(), type.getShape(), type.getElementType(),
        type.getEncoding());
    auto gathered = cast<linalg::GenericOp>(
        mlir::clone(rewriter, producer, TypeRange{type}, ValueRange{output}));
    for (auto index : gathered.getRegion().front().getOps<linalg::IndexOp>())
      index.setDim(inverse[index.getDim()]);
    rewriter.replaceOp(transpose, gathered->getResults());
    rewriter.eraseOp(producer);
    return success();
  }
};
} // namespace

void mlir::buddy::populateGatherPatterns(RewritePatternSet &patterns) {
  patterns.add<FoldGatherSlice, FoldGatherTranspose>(patterns.getContext());
}

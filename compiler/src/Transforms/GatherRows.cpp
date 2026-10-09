#include "Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
class GatherRows : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics() || op.getNumDpsInputs() != 1 ||
        op.getNumDpsInits() != 1 || op.getNumLoops() != 3 ||
        op.getNumParallelLoops() != 3)
      return failure();
    auto maps = op.getIndexingMapsArray();
    auto *context = op.getContext();
    if (maps[0] != AffineMap::get(3, 0,
                                  {getAffineDimExpr(0, context),
                                   getAffineDimExpr(1, context)},
                                  context) ||
        !maps[1].isIdentity())
      return failure();
    Block &body = op.getRegion().front();
    if (body.getOperations().size() != 4 || !body.getArgument(1).use_empty())
      return failure();
    auto yield = cast<linalg::YieldOp>(body.getTerminator());
    auto load = yield.getValues()[0].getDefiningOp<memref::LoadOp>();
    if (!load || load.getIndices().size() != 3 ||
        load.getMemRef().getParentBlock() == &body ||
        load.getIndices()[0].getParentBlock() == &body)
      return failure();
    auto row = load.getIndices()[1].getDefiningOp<arith::IndexCastOp>();
    auto feature = load.getIndices()[2].getDefiningOp<linalg::IndexOp>();
    if (!row || row.getIn() != body.getArgument(0) || !feature ||
        feature.getDim() != 2)
      return failure();
    Value indices = op.getDpsInputs()[0], output = op.getDpsInits()[0];
    auto tableType = load.getMemRefType();
    auto outputType = cast<MemRefType>(output.getType());
    auto indicesType = cast<MemRefType>(indices.getType());
    SmallVector<int64_t> tableStrides, outputStrides;
    int64_t offset;
    if (!output.getDefiningOp<memref::AllocOp>() ||
        indicesType.getRank() != 2 || tableType.getRank() != 3 ||
        outputType.getRank() != 3 || outputType.isDynamicDim(2) ||
        tableType.getDimSize(2) != outputType.getDimSize(2) ||
        tableType.getElementType() != outputType.getElementType() ||
        failed(tableType.getStridesAndOffset(tableStrides, offset)) ||
        failed(outputType.getStridesAndOffset(outputStrides, offset)) ||
        tableStrides[2] != 1 || outputStrides[2] != 1)
      return failure();

    for (Value input : {load.getMemRef(), indices}) {
      while (auto view = input.getDefiningOp<ViewLikeOpInterface>())
        input = view.getViewSource();
      if (input == output)
        return failure();
      if (auto argument = dyn_cast<BlockArgument>(input)) {
        if (argument.getOwner() == &body ||
            !isa<func::FuncOp>(argument.getOwner()->getParentOp()))
          return failure();
      } else if (!input.getDefiningOp<memref::AllocOp>()) {
        return failure();
      }
    }

    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value batches = rewriter.create<memref::DimOp>(loc, output, 0);
    Value tokens = rewriter.create<memref::DimOp>(loc, output, 1);
    int64_t width = outputType.getDimSize(2);
    auto rowType = MemRefType::get(
        {width}, outputType.getElementType(),
        StridedLayoutAttr::get(context, ShapedType::kDynamic, {1}),
        outputType.getMemorySpace());
    auto batchLoop = rewriter.create<scf::ForOp>(loc, zero, batches, one);
    rewriter.setInsertionPointToStart(batchLoop.getBody());
    auto tokenLoop = rewriter.create<scf::ForOp>(loc, zero, tokens, one);
    rewriter.setInsertionPointToStart(tokenLoop.getBody());
    Value index = rewriter.create<memref::LoadOp>(
        loc, indices,
        ValueRange{batchLoop.getInductionVar(), tokenLoop.getInductionVar()});
    Value selected = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), index);
    SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(1),
                                    rewriter.getIndexAttr(width)};
    SmallVector<OpFoldResult> strides(3, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sourceOffsets{load.getIndices()[0], selected,
                                            rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> outputOffsets{batchLoop.getInductionVar(),
                                            tokenLoop.getInductionVar(),
                                            rewriter.getIndexAttr(0)};
    auto sourceRowType =
        MemRefType::get({width}, tableType.getElementType(),
                        rowType.getLayout(), tableType.getMemorySpace());
    Value source = rewriter.create<memref::SubViewOp>(
        loc, sourceRowType, load.getMemRef(), sourceOffsets, sizes, strides);
    Value destination = rewriter.create<memref::SubViewOp>(
        loc, rowType, output, outputOffsets, sizes, strides);
    rewriter.create<memref::CopyOp>(loc, source, destination);
    rewriter.eraseOp(op);
    return success();
  }
};

class ScatterRows : public OpRewritePattern<vector::TransferWriteOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp write,
                                PatternRewriter &rewriter) const override {
    auto read = write.getVector().getDefiningOp<vector::TransferReadOp>();
    auto sourceType = read ? dyn_cast<MemRefType>(read.getBase().getType())
                           : MemRefType();
    auto destinationType = dyn_cast<MemRefType>(write.getBase().getType());
    if (!read || !sourceType || !destinationType ||
        !read->hasOneUse() || read->getNextNode() != write.getOperation() ||
        read.getMask() || write.getMask() ||
        read.getVectorType().getRank() != 1 ||
        read.getVectorType().isScalable() || !read.isDimInBounds(0) ||
        !write.isDimInBounds(0) ||
        !read.getPermutationMap().isMinorIdentity() ||
        !write.getPermutationMap().isMinorIdentity())
      return failure();

    int64_t width = read.getVectorType().getDimSize(0);
    SmallVector<int64_t> sourceStrides, destinationStrides;
    int64_t offset;
    if (sourceType.getRank() < 1 || destinationType.getRank() < 1 ||
        sourceType.getShape().back() != width ||
        destinationType.getShape().back() != width ||
        sourceType.getElementType() != read.getVectorType().getElementType() ||
        destinationType.getElementType() != read.getVectorType().getElementType() ||
        !matchPattern(read.getIndices().back(), m_Zero()) ||
        !matchPattern(write.getIndices().back(), m_Zero()) ||
        failed(sourceType.getStridesAndOffset(sourceStrides, offset)) ||
        failed(destinationType.getStridesAndOffset(destinationStrides, offset)) ||
        sourceStrides.back() != 1 || destinationStrides.back() != 1)
      return failure();

    Value sourceRoot = read.getBase(), destinationRoot = write.getBase();
    while (auto view = sourceRoot.getDefiningOp<ViewLikeOpInterface>())
      sourceRoot = view.getViewSource();
    while (auto view = destinationRoot.getDefiningOp<ViewLikeOpInterface>())
      destinationRoot = view.getViewSource();
    if (!destinationRoot.getDefiningOp<memref::AllocOp>() ||
        sourceRoot == destinationRoot)
      return failure();
    if (auto argument = dyn_cast<BlockArgument>(sourceRoot)) {
      if (!isa<func::FuncOp>(argument.getOwner()->getParentOp()))
        return failure();
    } else if (!sourceRoot.getDefiningOp<memref::AllocOp>()) {
      return failure();
    }

    SmallVector<Value> rows;
    for (auto transfer : {cast<VectorTransferOpInterface>(read.getOperation()),
                          cast<VectorTransferOpInterface>(write.getOperation())}) {
      auto type = cast<MemRefType>(transfer.getBase().getType());
      auto rowType = MemRefType::get(
          {width}, type.getElementType(),
          StridedLayoutAttr::get(getContext(), ShapedType::kDynamic, {1}),
          type.getMemorySpace());
      SmallVector<OpFoldResult> offsets;
      for (Value index : transfer.getIndices())
        offsets.push_back(index);
      SmallVector<OpFoldResult> sizes(type.getRank(), rewriter.getIndexAttr(1));
      sizes.back() = rewriter.getIndexAttr(width);
      SmallVector<OpFoldResult> strides(type.getRank(), rewriter.getIndexAttr(1));
      rows.push_back(rewriter.create<memref::SubViewOp>(
          write.getLoc(), rowType, transfer.getBase(), offsets, sizes, strides));
    }
    rewriter.create<memref::CopyOp>(write.getLoc(), rows[0], rows[1]);
    rewriter.eraseOp(write);
    rewriter.eraseOp(read);
    return success();
  }
};

class GatherRowsPass
    : public PassWrapper<GatherRowsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GatherRowsPass)
  StringRef getArgument() const final { return "lower-gather-rows"; }
  StringRef getDescription() const final {
    return "Copy contiguous gathered and scattered rows without element loops.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<GatherRows, ScatterRows>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::buddy::registerGatherRowsPass() {
  PassRegistration<GatherRowsPass>();
}

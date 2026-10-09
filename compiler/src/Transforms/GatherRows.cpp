#include "Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

class GatherRowsPass
    : public PassWrapper<GatherRowsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GatherRowsPass)
  StringRef getArgument() const final { return "lower-gather-rows"; }
  StringRef getDescription() const final {
    return "Copy contiguous gathered rows without an element loop.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<GatherRows>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::buddy::registerGatherRowsPass() {
  PassRegistration<GatherRowsPass>();
}

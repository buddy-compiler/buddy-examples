#include "Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
// Keep the original add on one KV head; the runtime repeats its descriptor.
Value singleHeadRhs(Value rhs, int64_t batches, bool &transpose,
                    PatternRewriter &rewriter) {
  auto collapse = rhs.getDefiningOp<tensor::CollapseShapeOp>();
  if (!collapse || !rhs.hasOneUse() ||
      collapse.getReassociationIndices() !=
          SmallVector<ReassociationIndices>{{0, 1}, {2}, {3}})
    return {};
  auto broadcast = collapse.getSrc().getDefiningOp<linalg::GenericOp>();
  if (!broadcast || !broadcast->hasOneUse() || broadcast.getLibraryCallAttr() ||
      broadcast.getInputs().empty() || broadcast.getInputs().size() > 2 ||
      broadcast.getOutputs().size() != 1 || broadcast.getNumLoops() != 4 ||
      broadcast.getNumParallelLoops() != 4 ||
      !llvm::hasSingleElement(
          broadcast.getRegion().front().without_terminator()))
    return {};
  auto type = dyn_cast<RankedTensorType>(broadcast.getResult(0).getType());
  if (!type || !type.hasStaticShape() || type.getShape()[0] != 1 ||
      type.getShape()[1] != batches)
    return {};
  auto add = dyn_cast<arith::AddFOp>(broadcast.getRegion().front().front());
  auto &body = broadcast.getRegion().front();
  if (!add || body.getTerminator()->getOperand(0) != add.getResult())
    return {};
  auto identity = AffineMap::getMultiDimIdentityMap(4, rewriter.getContext());
  auto maps = broadcast.getIndexingMapsArray();
  auto headMap = AffineMap::get(
      4, 0,
      {rewriter.getAffineDimExpr(0), rewriter.getAffineConstantExpr(0),
       rewriter.getAffineDimExpr(2), rewriter.getAffineDimExpr(3)},
      rewriter.getContext());
  unsigned sourceIndex = 0, zeroIndex = 1;
  APFloat zeroValue(0.0f);
  if (broadcast.getInputs().size() == 2) {
    sourceIndex = maps[0] == headMap ? 0 : 1;
    zeroIndex = 1 - sourceIndex;
    if (add.getLhs() != body.getArgument(0) ||
        add.getRhs() != body.getArgument(1) || maps[sourceIndex] != headMap ||
        maps[zeroIndex] != identity || maps[2] != identity)
      return {};
    auto zero =
        broadcast.getInputs()[zeroIndex].getDefiningOp<arith::ConstantOp>();
    auto values = zero ? dyn_cast<DenseFPElementsAttr>(zero.getValue())
                       : DenseFPElementsAttr{};
    if (!values || !values.isSplat())
      return {};
    zeroValue = values.getSplatValue<APFloat>();
  } else {
    bool sourceFirst = add.getLhs() == body.getArgument(0);
    if ((!sourceFirst && add.getRhs() != body.getArgument(0)) ||
        maps[0] != headMap || maps[1] != identity)
      return {};
    auto zero = (sourceFirst ? add.getRhs() : add.getLhs())
                    .getDefiningOp<arith::ConstantOp>();
    auto value = zero ? dyn_cast<FloatAttr>(zero.getValue()) : FloatAttr{};
    if (!value)
      return {};
    zeroValue = value.getValue();
    zeroIndex = sourceFirst ? 1 : 0;
  }
  if (!zeroValue.isZero() || zeroValue.isNegative())
    return {};
  Value source = broadcast.getInputs()[sourceIndex];
  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  if (!sourceType || !sourceType.hasStaticShape() ||
      sourceType.getRank() != 4 || sourceType.getShape()[0] != 1 ||
      sourceType.getShape()[1] != 1 ||
      sourceType.getShape()[2] != type.getShape()[2] ||
      sourceType.getShape()[3] != type.getShape()[3])
    return {};
  if (auto op = source.getDefiningOp<linalg::TransposeOp>()) {
    if (!source.hasOneUse() ||
        op.getPermutation() != ArrayRef<int64_t>({0, 1, 3, 2}))
      return {};
    source = op.getInput();
    sourceType = cast<RankedTensorType>(source.getType());
    transpose = !transpose;
  }
  auto loc = broadcast.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(broadcast);
  Value empty = rewriter.create<tensor::EmptyOp>(loc, sourceType.getShape(),
                                                 sourceType.getElementType());
  auto normalized = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{sourceType}, ValueRange{source}, ValueRange{empty},
      ArrayRef<AffineMap>{identity, identity},
      SmallVector<utils::IteratorType>(4, utils::IteratorType::parallel),
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        Value zero = builder.create<arith::ConstantOp>(
            loc, builder.getFloatAttr(sourceType.getElementType(), zeroValue));
        auto sameAdd =
            builder.create<arith::AddFOp>(loc, zeroIndex == 0 ? zero : args[0],
                                          zeroIndex == 0 ? args[0] : zero);
        sameAdd->setAttrs(add->getAttrs());
        builder.create<linalg::YieldOp>(loc, sameAdd.getResult());
      });
  return normalized.getResult(0);
}

template <typename Op> struct Outline : OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    if (!getElementTypeOrSelf(op.getInputs()[0].getType()).isF32() ||
        !getElementTypeOrSelf(op.getInputs()[1].getType()).isF32() ||
        !getElementTypeOrSelf(op.getOutputs()[0].getType()).isF32())
      return failure();
    const bool batch = isa<linalg::BatchMatmulOp>(op.getOperation());
    unsigned shift = batch ? 1 : 0;
    auto ctx = rewriter.getContext();
    auto m = getAffineDimExpr(shift, ctx);
    auto n = getAffineDimExpr(shift + 1, ctx);
    auto k = getAffineDimExpr(shift + 2, ctx);
    SmallVector<AffineExpr> lhs, rhs, transposed, output;
    if (batch) {
      auto b = getAffineDimExpr(0, ctx);
      lhs.push_back(b);
      rhs.push_back(b);
      transposed.push_back(b);
      output.push_back(b);
    }
    lhs.append({m, k});
    rhs.append({k, n});
    transposed.append({n, k});
    output.append({m, n});
    auto map = [&](ArrayRef<AffineExpr> results) {
      return AffineMap::get(shift + 3, 0, results, ctx);
    };
    auto maps = op.getIndexingMapsArray();
    if (maps[0] != map(lhs) || maps[2] != map(output))
      return failure();
    bool rhsTransposed = maps[1] == map(transposed);
    if (!rhsTransposed && maps[1] != map(rhs))
      return failure();
    SmallVector<Value> inputs(op.getInputs());
    StringRef callee =
        rhsTransposed ? "rvv_matmul_transpose_rhs" : "rvv_matmul";
    if (batch) {
      auto type = dyn_cast<RankedTensorType>(inputs[0].getType());
      if (type && type.hasStaticShape()) {
        if (Value single = singleHeadRhs(inputs[1], type.getShape()[0],
                                         rhsTransposed, rewriter)) {
          inputs[1] = single;
          maps[1] =
              AffineMap::get(4, 0,
                             {rewriter.getAffineConstantExpr(0),
                              rewriter.getAffineConstantExpr(0),
                              rhsTransposed ? n : k, rhsTransposed ? k : n},
                             ctx);
          callee = rhsTransposed ? "rvv_matmul_broadcast_transpose_rhs"
                                 : "rvv_matmul_broadcast_rhs";
        }
      }
    }
    auto call = rewriter.create<linalg::GenericOp>(
        op.getLoc(), op.getResultTypes(), inputs, op.getOutputs(), maps,
        op.getIteratorTypesArray(),
        [](OpBuilder &builder, Location loc, ValueRange args) {
          Value product = builder.create<arith::MulFOp>(loc, args[0], args[1]);
          Value sum = builder.create<arith::AddFOp>(loc, args[2], product);
          builder.create<linalg::YieldOp>(loc, sum);
        });
    call.setLibraryCallAttr(rewriter.getStringAttr(callee));
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
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
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

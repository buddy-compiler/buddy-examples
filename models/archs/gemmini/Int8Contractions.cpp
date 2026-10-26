#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Config/llvm-config.h"

using namespace mlir;

namespace {
class NativeInt8ContractionLowering
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &b) const override {
    bool matrix = op->hasAttr("buddy.int8_matmul");
    auto conv = op->getAttrOfType<DictionaryAttr>("buddy.int8_conv");
    if (!matrix && !conv)
      return failure();
    Value input = op.getInputs()[0], weight = op.getInputs()[1];
    Value output = op.getOutputs()[0];
    auto in = dyn_cast<MemRefType>(input.getType());
    auto wt = dyn_cast<MemRefType>(weight.getType());
    auto out = dyn_cast<MemRefType>(output.getType());
    if (!in || !wt || !out || !in.hasStaticShape() || !wt.hasStaticShape() ||
        !out.hasStaticShape() || !in.getElementType().isInteger(8) ||
        !wt.getElementType().isInteger(8) ||
        !out.getElementType().isInteger(32))
      return op.emitError(
          "Gemmini requires static signed INT8 contractions with INT32 output");
    Location loc = op.getLoc();
    if (matrix) {
      linalg::MatmulOp::create(b, loc, ValueRange{input, weight},
                               ValueRange{output});
      b.eraseOp(op);
      return success();
    }
    int64_t groups = cast<IntegerAttr>(conv.get("groups")).getInt();
    auto order =
        cast<DenseI64ArrayAttr>(conv.get("weight_permutation")).asArrayRef();
    auto stride = cast<DenseI64ArrayAttr>(conv.get("stride")).asArrayRef();
    auto dilation = cast<DenseI64ArrayAttr>(conv.get("dilation")).asArrayRef();
    auto maps = op.getIndexingMapsArray();
    bool inputNhwc = maps[0].getResult(3).isFunctionOfDim(4);
    bool outputNhwc =
        maps[2].getResult(3) == getAffineDimExpr(1, b.getContext());
    SmallVector<int64_t> logicalWeight(4);
    for (auto [axis, logical] : llvm::enumerate(order))
      logicalWeight[logical] = wt.getShape()[axis];
    int64_t channels = logicalWeight[0], cin = logicalWeight[1];
    int64_t kh = logicalWeight[2], kw = logicalWeight[3];
    int64_t oh = out.getShape()[outputNhwc ? 1 : 2];
    int64_t ow = out.getShape()[outputNhwc ? 2 : 3];
    int64_t rows = out.getShape()[0] * oh * ow, reduction = cin * kh * kw;
    int64_t columns = channels / groups;
    auto idx = [&](int64_t x) -> Value {
      return arith::ConstantIndexOp::create(b, loc, x);
    };
    Value zero = idx(0), one = idx(1);
    Value a = memref::AllocOp::create(
        b, loc, MemRefType::get({rows, reduction}, b.getI8Type()));
    Value rhs = memref::AllocOp::create(
        b, loc, MemRefType::get({reduction, columns}, b.getI8Type()));
    Value c = memref::AllocOp::create(
        b, loc, MemRefType::get({rows, columns}, b.getI32Type()));
    auto grouped = scf::ForOp::create(b, loc, zero, idx(groups), one);
    b.setInsertionPointToStart(grouped.getBody());
    Value group = grouped.getInductionVar();
    auto kk = scf::ForOp::create(b, loc, zero, idx(reduction), one);
    b.setInsertionPointToStart(kk.getBody());
    auto jj = scf::ForOp::create(b, loc, zero, idx(columns), one);
    b.setInsertionPointToStart(jj.getBody());
    Value k = kk.getInductionVar(), j = jj.getInductionVar();
    SmallVector<Value> logicalIndices{
        arith::AddIOp::create(
            b, loc, arith::MulIOp::create(b, loc, group, idx(columns)), j),
        arith::DivUIOp::create(b, loc, k, idx(kh * kw)),
        arith::RemUIOp::create(
            b, loc, arith::DivUIOp::create(b, loc, k, idx(kw)), idx(kh)),
        arith::RemUIOp::create(b, loc, k, idx(kw))};
    SmallVector<Value> storedIndices;
    for (int64_t axis : order)
      storedIndices.push_back(logicalIndices[axis]);
    Value w = memref::LoadOp::create(b, loc, weight, storedIndices);
    memref::StoreOp::create(b, loc, w, rhs, ValueRange{k, j});
    b.setInsertionPointAfter(kk);
    auto row = scf::ForOp::create(b, loc, zero, idx(rows), one);
    b.setInsertionPointToStart(row.getBody());
    auto column = scf::ForOp::create(b, loc, zero, idx(reduction), one);
    b.setInsertionPointToStart(column.getBody());
    Value r = row.getInductionVar();
    k = column.getInductionVar();
    Value batch = arith::DivUIOp::create(b, loc, r, idx(oh * ow));
    Value y = arith::RemUIOp::create(
        b, loc, arith::DivUIOp::create(b, loc, r, idx(ow)), idx(oh));
    Value x = arith::RemUIOp::create(b, loc, r, idx(ow));
    Value ci = arith::AddIOp::create(
        b, loc, arith::MulIOp::create(b, loc, group, idx(cin)),
        arith::DivUIOp::create(b, loc, k, idx(kh * kw)));
    Value ky = arith::RemUIOp::create(
        b, loc, arith::DivUIOp::create(b, loc, k, idx(kw)), idx(kh));
    Value kx = arith::RemUIOp::create(b, loc, k, idx(kw));
    y = arith::AddIOp::create(
        b, loc, arith::MulIOp::create(b, loc, y, idx(stride[0])),
        arith::MulIOp::create(b, loc, ky, idx(dilation[0])));
    x = arith::AddIOp::create(
        b, loc, arith::MulIOp::create(b, loc, x, idx(stride[1])),
        arith::MulIOp::create(b, loc, kx, idx(dilation[1])));
    Value pixel = memref::LoadOp::create(
        b, loc, input,
        inputNhwc ? ValueRange{batch, y, x, ci} : ValueRange{batch, ci, y, x});
    memref::StoreOp::create(b, loc, pixel, a, ValueRange{r, k});
    b.setInsertionPointAfter(row);
    Value z = arith::ConstantIntOp::create(b, loc, 0, 32);
    linalg::FillOp::create(b, loc, z, c);
    linalg::MatmulOp::create(b, loc, ValueRange{a, rhs}, ValueRange{c});
    auto rr = scf::ForOp::create(b, loc, zero, idx(rows), one);
    b.setInsertionPointToStart(rr.getBody());
    auto cc = scf::ForOp::create(b, loc, zero, idx(columns), one);
    b.setInsertionPointToStart(cc.getBody());
    Value i = rr.getInductionVar();
    j = cc.getInductionVar();
    Value value = memref::LoadOp::create(b, loc, c, ValueRange{i, j});
    Value n = arith::DivUIOp::create(b, loc, i, idx(oh * ow));
    Value channel = arith::AddIOp::create(
        b, loc, arith::MulIOp::create(b, loc, group, idx(columns)), j);
    Value yy = arith::RemUIOp::create(
        b, loc, arith::DivUIOp::create(b, loc, i, idx(ow)), idx(oh));
    Value xx = arith::RemUIOp::create(b, loc, i, idx(ow));
    memref::StoreOp::create(b, loc, value, output,
                            outputNhwc ? ValueRange{n, yy, xx, channel}
                                       : ValueRange{n, channel, yy, xx});
    b.setInsertionPointAfter(grouped);
    memref::DeallocOp::create(b, loc, a);
    memref::DeallocOp::create(b, loc, rhs);
    memref::DeallocOp::create(b, loc, c);
    b.eraseOp(op);
    return success();
  }
};

struct PrepareGemminiInt8Pass
    : PassWrapper<PrepareGemminiInt8Pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareGemminiInt8Pass)
  StringRef getArgument() const final { return "prepare-gemmini-int8"; }
  StringRef getDescription() const final {
    return "Prepare project INT8 contractions for Gemmini lowering";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<NativeInt8ContractionLowering>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "GemminiModel", LLVM_VERSION_STRING,
          [] { PassRegistration<PrepareGemminiInt8Pass>(); }};
}

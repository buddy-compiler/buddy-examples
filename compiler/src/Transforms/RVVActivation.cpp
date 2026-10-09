#include "RVVTensorKernels.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
namespace mlir::buddy::rvv {
namespace {
struct Unary : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &r) const override {
    if (op.getNumResults() != 1 || op.getNumDpsInputs() != 1 ||
        op.getLibraryCallAttr() || op.getNumParallelLoops() != op.getNumLoops())
      return failure();
    auto type = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!type || !type.hasStaticShape() || !type.getElementType().isF32() ||
        op.getInputs()[0].getType() != type)
      return failure();
    for (AffineMap map : op.getIndexingMapsArray())
      if (!map.isIdentity())
        return failure();
    auto yield = cast<linalg::YieldOp>(op.getBody()->getTerminator());
    Operation *unary = yield.getValues()[0].getDefiningOp();
    StringRef callee;
    if (isa_and_nonnull<math::TanhOp>(unary))
      callee = "rvv_tanh";
    else if (isa_and_nonnull<math::SinOp>(unary))
      callee = "rvv_sin";
    else if (isa_and_nonnull<math::CosOp>(unary))
      callee = "rvv_cos";
    if (callee.empty() ||
        unary->getOperand(0) != op.getBody()->getArgument(0) ||
        std::distance(op.getBody()->begin(), op.getBody()->end()) != 2)
      return failure();
    Location loc = op.getLoc();
    Value output = r.create<memref::AllocOp>(
        loc, MemRefType::get(type.getShape(), type.getElementType()));
    auto unranked = UnrankedMemRefType::get(type.getElementType(), 0);
    Value out = r.create<memref::CastOp>(loc, unranked, output);
    auto layout = StridedLayoutAttr::get(
        r.getContext(), ShapedType::kDynamic,
        SmallVector<int64_t>(type.getRank(), ShapedType::kDynamic));
    Value input = r.create<bufferization::ToBufferOp>(
        loc, MemRefType::get(type.getShape(), type.getElementType(), layout),
        op.getInputs()[0], r.getUnitAttr());
    Value in = r.create<memref::CastOp>(loc, unranked, input);
    auto module = op->getParentOfType<ModuleOp>();
    if (!module.lookupSymbol<func::FuncOp>(callee)) {
      OpBuilder::InsertionGuard guard(r);
      r.setInsertionPointToStart(module.getBody());
      auto fn = r.create<func::FuncOp>(
          loc, callee, r.getFunctionType({unranked, unranked}, {}));
      fn.setPrivate();
      fn->setAttr("llvm.emit_c_interface", r.getUnitAttr());
    }
    r.create<func::CallOp>(loc, callee, TypeRange{}, ValueRange{out, in});
    Value tensor = r.create<bufferization::ToTensorOp>(
        loc, type, output, r.getUnitAttr(), r.getUnitAttr());
    r.replaceOp(op, tensor);
    return success();
  }
};
} // namespace
void populateActivationPatterns(RewritePatternSet &patterns) {
  patterns.add<Unary>(patterns.getContext());
}
} // namespace mlir::buddy::rvv

#include "Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include <functional>

using namespace mlir;
namespace {
class Pointwise : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &r) const override {
    if (!op.hasPureBufferSemantics() || op.getNumDpsInits() != 1 ||
        op.getNumParallelLoops() != op.getNumLoops() || op.getLibraryCallAttr())
      return failure();
    auto out = op.getDpsInits()[0];
    auto type = cast<MemRefType>(out.getType());
    auto maps = op.getIndexingMapsArray();
    if (!type.getElementType().isF32() || !type.hasStaticShape() ||
        !maps.back().isIdentity() || type.getRank() != op.getNumLoops())
      return failure();
    auto *body = op.getBody();
    Value result = cast<linalg::YieldOp>(body->getTerminator()).getValues()[0];
    auto opcode = [](Operation *expression) -> int32_t {
      if (isa<arith::AddFOp>(expression))
        return 0;
      if (isa<arith::SubFOp>(expression))
        return 1;
      if (isa<arith::MulFOp>(expression))
        return 2;
      if (isa<arith::DivFOp>(expression))
        return 3;
      if (isa<arith::NegFOp>(expression))
        return 4;
      if (isa<math::AbsFOp>(expression))
        return 5;
      if (isa<arith::MaximumFOp>(expression))
        return 6;
      if (isa<arith::MinimumFOp>(expression))
        return 7;
      return -1;
    };
    SmallVector<Operation *> expressions;
    SmallVector<Value> constants;
    DenseMap<Value, Value> buffers;
    llvm::SmallDenseSet<Value, 8> visited;
    std::function<LogicalResult(Value)> collect = [&](Value value) {
      if (!visited.insert(value).second)
        return success();
      if (auto argument = dyn_cast<BlockArgument>(value)) {
        if (argument.getOwner() != body ||
            argument.getArgNumber() >= op.getNumDpsInputs())
          return failure();
        unsigned i = argument.getArgNumber();
        auto input = op.getDpsInputs()[i];
        auto inputType = dyn_cast<MemRefType>(input.getType());
        if (!inputType || !inputType.getElementType().isF32() ||
            !inputType.hasStaticShape() || inputType.getRank() > type.getRank())
          return failure();
        auto map = maps[i];
        for (int64_t axis = 0; axis < inputType.getRank(); ++axis) {
          int64_t outputAxis = type.getRank() - inputType.getRank() + axis;
          if (auto dim = dyn_cast<AffineDimExpr>(map.getResult(axis))) {
            if (dim.getPosition() != outputAxis ||
                (inputType.getDimSize(axis) != 1 &&
                 inputType.getDimSize(axis) != type.getDimSize(outputAxis)))
              return failure();
          } else if (auto zero =
                         dyn_cast<AffineConstantExpr>(map.getResult(axis))) {
            if (zero.getValue() != 0 || inputType.getDimSize(axis) != 1)
              return failure();
          } else
            return failure();
        }
        buffers[value] = input;
        return success();
      }
      if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
        if (!constant.getType().isF32())
          return failure();
        constants.push_back(value);
        return success();
      }
      Operation *expression = value.getDefiningOp();
      if (!expression || expression->getBlock() != body ||
          opcode(expression) < 0 || !value.getType().isF32())
        return failure();
      for (Value operand : expression->getOperands())
        if (failed(collect(operand)))
          return failure();
      expressions.push_back(expression);
      return success();
    };
    if (failed(collect(result)) || expressions.empty())
      return failure();
    Location loc = op.getLoc();
    for (Value value : constants) {
      auto constant = value.getDefiningOp<arith::ConstantOp>();
      Value outside = r.create<arith::ConstantOp>(loc, constant.getValue());
      Value buffer =
          r.create<memref::AllocaOp>(loc, MemRefType::get({}, r.getF32Type()));
      r.create<memref::StoreOp>(loc, outside, buffer, ValueRange{});
      buffers[value] = buffer;
    }
    auto module = op->getParentOfType<ModuleOp>();
    for (Operation *expression : expressions) {
      int32_t code = opcode(expression);
      Value destination =
          expression->getResult(0) == result
              ? out
              : r.create<memref::AllocOp>(
                     loc,
                     MemRefType::get(type.getShape(), type.getElementType(),
                                     MemRefLayoutAttrInterface{},
                                     type.getMemorySpace()))
                    .getResult();
      SmallVector<Value> arguments;
      auto castBuffer = [&](Value buffer) -> Value {
        auto t = cast<MemRefType>(buffer.getType());
        return r.create<memref::CastOp>(
            loc, UnrankedMemRefType::get(r.getF32Type(), t.getMemorySpace()),
            buffer);
      };
      arguments.push_back(castBuffer(destination));
      for (Value operand : expression->getOperands())
        arguments.push_back(castBuffer(buffers.lookup(operand)));
      arguments.push_back(r.create<arith::ConstantIntOp>(loc, code, 32));
      StringRef name =
          expression->getNumOperands() == 2 ? "rvv_binary" : "rvv_unary";
      if (!module.lookupSymbol<func::FuncOp>(name)) {
        OpBuilder::InsertionGuard guard(r);
        r.setInsertionPointToStart(module.getBody());
        SmallVector<Type> types;
        for (Value value : arguments)
          types.push_back(value.getType());
        auto function =
            r.create<func::FuncOp>(loc, name, r.getFunctionType(types, {}));
        function.setPrivate();
        function->setAttr("llvm.emit_c_interface", r.getUnitAttr());
      }
      r.create<func::CallOp>(loc, name, TypeRange{}, arguments);
      buffers[expression->getResult(0)] = destination;
    }
    r.eraseOp(op);
    return success();
  }
};
class LowerRVVPointwisePass
    : public PassWrapper<LowerRVVPointwisePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerRVVPointwisePass)
  StringRef getArgument() const final { return "lower-rvv-pointwise"; }
  StringRef getDescription() const final {
    return "Lower bank-backed FP32 pointwise arithmetic to the NPU";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                math::MathDialect, memref::MemRefDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<Pointwise>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
void mlir::buddy::registerRVVPointwisePass() {
  PassRegistration<LowerRVVPointwisePass>();
}

#include "RVVTensorKernels.h"
#include "Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
using namespace mlir::buddy::rvv;
Value silu(Value value) {
  auto multiply = operation(value, "arith.mulf");
  if (!multiply)
    return {};
  Value input = source(multiply->getOperand(0));
  auto reciprocal = operation(multiply->getOperand(1), "arith.divf");
  if (!reciprocal || !constant(reciprocal->getOperand(0), 1.0))
    return {};
  auto add = operation(reciprocal->getOperand(1), "arith.addf");
  if (!add || !constant(add->getOperand(1), 1.0))
    return {};
  auto exponential = operation(add->getOperand(0), "math.exp");
  if (!exponential)
    return {};
  auto negative = operation(exponential->getOperand(0), "arith.negf");
  if (!negative || source(negative->getOperand(0)) != input)
    return {};
  return input;
}

Value tensorInput(Value origin, linalg::GenericOp op, RankedTensorType type) {
  if (!origin)
    return {};
  for (Value input : op.getInputs())
    if (input.getType() == type && source(input) == origin)
      return input;
  return {};
}

struct Outline : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1 || op.getLibraryCallAttr())
      return failure();
    auto type = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!type || !type.hasStaticShape() || !type.getElementType().isF32())
      return failure();
    StringRef kernel;
    SmallVector<Value> inputs;
    if (auto multiply = operation(op.getResult(0), "arith.mulf")) {
      if (Value gate = tensorInput(silu(multiply->getOperand(0)), op, type)) {
        Value up = tensorInput(source(multiply->getOperand(1)), op, type);
        if (up) {
          kernel = "rvv_swiglu";
          inputs = {gate, up};
        }
      }
    }
    if (kernel.empty()) {
      if (Value input = tensorInput(silu(op.getResult(0)), op, type)) {
        for (Operation *user : op.getResult(0).getUsers()) {
          auto consumer = dyn_cast<linalg::GenericOp>(user);
          if (!consumer || consumer.getNumResults() != 1)
            continue;
          auto multiply = operation(consumer.getResult(0), "arith.mulf");
          if (multiply && silu(multiply->getOperand(0)) &&
              tensorInput(source(multiply->getOperand(1)), consumer, type))
            return failure();
        }
        kernel = "rvv_silu";
        inputs = {input};
      }
    }
    if (kernel.empty() && type.getRank() == 3 && type.getShape()[0] == 1) {
      auto add = operation(op.getResult(0), "arith.addf");
      if (!add)
        return failure();
      Value hidden = source(add->getOperand(0));
      auto multiply = operation(add->getOperand(1), "arith.mulf");
      if (!multiply)
        return failure();
      auto square = operation(multiply->getOperand(0), "math.fpowi");
      auto reciprocal = operation(multiply->getOperand(1), "arith.divf");
      if (!square || !constant(square->getOperand(1), 2.0) || !reciprocal ||
          !constant(reciprocal->getOperand(0), 1.0))
        return failure();
      auto sine = operation(square->getOperand(0), "math.sin");
      auto denominator = operation(reciprocal->getOperand(1), "arith.addf");
      if (!sine || !denominator ||
          !constant(denominator->getOperand(1), double(float(1e-9))))
        return failure();
      auto scaled = operation(sine->getOperand(0), "arith.mulf");
      auto beta = operation(denominator->getOperand(0), "math.exp");
      if (!scaled || !beta || source(scaled->getOperand(0)) != hidden)
        return failure();
      auto alpha = operation(scaled->getOperand(1), "math.exp");
      if (!alpha)
        return failure();
      Value logAlpha = source(alpha->getOperand(0));
      Value logBeta = source(beta->getOperand(0));
      auto coefficientType =
          RankedTensorType::get({type.getShape()[1]}, rewriter.getF32Type());
      if (hidden.getType() != type || logAlpha.getType() != coefficientType ||
          logBeta.getType() != coefficientType)
        return failure();
      kernel = "rvv_snake";
      inputs = {hidden, logAlpha, logBeta};
    }
    if (kernel.empty())
      return failure();
    SmallVector<AffineMap> maps;
    auto identity = AffineMap::getMultiDimIdentityMap(type.getRank(),
                                                      rewriter.getContext());
    for (Value input : inputs)
      maps.push_back(input.getType() == type
                         ? identity
                         : AffineMap::get(type.getRank(), 0,
                                          rewriter.getAffineDimExpr(1)));
    maps.push_back(identity);
    Value empty = rewriter.create<tensor::EmptyOp>(op.getLoc(), type.getShape(),
                                                   type.getElementType());
    auto call = rewriter.create<linalg::GenericOp>(
        op.getLoc(), TypeRange{type}, inputs, ValueRange{empty}, maps,
        SmallVector<utils::IteratorType>(type.getRank(),
                                         utils::IteratorType::parallel),
        [kernel](OpBuilder &builder, Location location, ValueRange args) {
          Value one = builder.create<arith::ConstantOp>(
              location, builder.getF32FloatAttr(1.0));
          Value result;
          if (kernel == "rvv_snake") {
            Value alpha = builder.create<math::ExpOp>(location, args[1]);
            Value beta = builder.create<math::ExpOp>(location, args[2]);
            Value scaled =
                builder.create<arith::MulFOp>(location, args[0], alpha);
            Value sine = builder.create<math::SinOp>(location, scaled);
            Value square = builder.create<arith::MulFOp>(location, sine, sine);
            Value epsilon = builder.create<arith::ConstantOp>(
                location, builder.getF32FloatAttr(float(1e-9)));
            Value denominator =
                builder.create<arith::AddFOp>(location, beta, epsilon);
            Value reciprocal =
                builder.create<arith::DivFOp>(location, one, denominator);
            Value fraction =
                builder.create<arith::MulFOp>(location, square, reciprocal);
            result = builder.create<arith::AddFOp>(location, args[0], fraction);
          } else {
            Value negative = builder.create<arith::NegFOp>(location, args[0]);
            Value exponential = builder.create<math::ExpOp>(location, negative);
            Value denominator =
                builder.create<arith::AddFOp>(location, exponential, one);
            Value reciprocal =
                builder.create<arith::DivFOp>(location, one, denominator);
            result =
                builder.create<arith::MulFOp>(location, args[0], reciprocal);
            if (kernel == "rvv_swiglu")
              result = builder.create<arith::MulFOp>(location, result, args[1]);
          }
          builder.create<linalg::YieldOp>(location, result);
        });
    call.setLibraryCallAttr(rewriter.getStringAttr(kernel));
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

class OutlineRVVKernelsPass
    : public PassWrapper<OutlineRVVKernelsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OutlineRVVKernelsPass)
  StringRef getArgument() const final { return "outline-rvv-kernels"; }
  StringRef getDescription() const final {
    return "Outline FP32 vector expressions into private RVV kernels.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, linalg::LinalgDialect, math::MathDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                memref::MemRefDialect, func::FuncDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<Outline>(&getContext());
    populateTensorPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

class LowerRVVKernelsPass
    : public PassWrapper<LowerRVVKernelsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerRVVKernelsPass)
  StringRef getArgument() const final { return "lower-rvv-kernels"; }
  StringRef getDescription() const final {
    return "Lower bufferized RVV kernels to runtime calls.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, memref::MemRefDialect>();
  }
  void runOnOperation() override {
    SmallVector<linalg::GenericOp> calls;
    getOperation().walk([&](linalg::GenericOp op) {
      if (auto name = op.getLibraryCallAttr();
          name &&
          (name.getValue() == "rvv_silu" || name.getValue() == "rvv_swiglu" ||
           name.getValue() == "rvv_snake" || name.getValue() == "rvv_matmul"))
        calls.push_back(op);
    });
    for (auto op : calls) {
      if (op.getNumResults() != 0) {
        op.emitError("RVV kernel must be bufferized before lowering");
        signalPassFailure();
        return;
      }
      OpBuilder builder(op);
      auto type = UnrankedMemRefType::get(builder.getF32Type(), 0);
      SmallVector<Value> arguments;
      arguments.push_back(builder.create<memref::CastOp>(
          op.getLoc(), type, op.getDpsInitOperand(0)->get()));
      for (Value input : op.getInputs())
        arguments.push_back(
            builder.create<memref::CastOp>(op.getLoc(), type, input));
      StringRef name = op.getLibraryCallAttr().getValue();
      if (!getOperation().lookupSymbol<func::FuncOp>(name)) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(getOperation().getBody());
        auto function = builder.create<func::FuncOp>(
            op.getLoc(), name,
            builder.getFunctionType(SmallVector<Type>(arguments.size(), type),
                                    TypeRange{}));
        function.setPrivate();
        function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
      }
      builder.create<func::CallOp>(op.getLoc(), name, TypeRange{}, arguments);
      op.erase();
    }
  }
};
} // namespace

void mlir::buddy::registerRVVKernelsPasses() {
  PassRegistration<OutlineRVVKernelsPass>();
  PassRegistration<LowerRVVKernelsPass>();
}

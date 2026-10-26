//===- LowerFrontend.cpp - The frontend operations in software ------------===//
//
// The frontend inserts fp32_to_int8, int8_to_fp32 and int32_to_fp32; a
// core's form-custom-ops recognizes supported computation patterns. This pass
// computes the others, on tensors, exactly as their reference lowering:
//
//   fp32_to_int8(x, multiplier) = clamp(round_half_even(x * multiplier),
//                                       -128, 127)
//   int8_to_fp32(q, step)      = q * step
//   int32_to_fp32(q, step)     = q * step
//
// The scale broadcasts along its dimensions of size one.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
// Reads the scale at the element of the result: index 0 where it broadcasts.
AffineMap broadcastMap(MLIRContext *context, RankedTensorType result,
                       RankedTensorType scale) {
  SmallVector<AffineExpr> results;
  for (int64_t dim = 0; dim < result.getRank(); ++dim)
    results.push_back(scale.getDimSize(dim) == 1 && result.getDimSize(dim) != 1
                          ? getAffineConstantExpr(0, context)
                          : getAffineDimExpr(dim, context));
  return AffineMap::get(result.getRank(), 0, results, context);
}

void lower(Operation *op, bool quantize) {
  OpBuilder builder(op);
  Location loc = op->getLoc();
  Value input = op->getOperand(0);
  Value scale = op->getOperand(1);
  auto result = cast<RankedTensorType>(op->getResult(0).getType());
  auto scaleType = cast<RankedTensorType>(scale.getType());
  MLIRContext *context = op->getContext();
  SmallVector<AffineMap> maps = {
      AffineMap::getMultiDimIdentityMap(result.getRank(), context),
      broadcastMap(context, result, scaleType),
      AffineMap::getMultiDimIdentityMap(result.getRank(), context)};
  SmallVector<utils::IteratorType> iterators(result.getRank(),
                                             utils::IteratorType::parallel);
  Value init = tensor::EmptyOp::create(builder, loc, result.getShape(),
                                       result.getElementType());
  auto generic = linalg::GenericOp::create(
      builder, loc, result, ValueRange{input, scale}, ValueRange{init}, maps,
      iterators, [&](OpBuilder &body, Location loc, ValueRange args) {
        Value value;
        if (quantize) {
          Value scaled = arith::MulFOp::create(body, loc, args[0], args[1]);
          Value rounded = math::RoundEvenOp::create(body, loc, scaled);
          Value low =
              arith::ConstantOp::create(body, loc, body.getF32FloatAttr(-128));
          Value high =
              arith::ConstantOp::create(body, loc, body.getF32FloatAttr(127));
          Value clamped = arith::MinimumFOp::create(
              body, loc, arith::MaximumFOp::create(body, loc, rounded, low),
              high);
          value = arith::FPToSIOp::create(body, loc, body.getI8Type(), clamped);
        } else {
          Value converted =
              arith::SIToFPOp::create(body, loc, body.getF32Type(), args[0]);
          value = arith::MulFOp::create(body, loc, converted, args[1]);
        }
        linalg::YieldOp::create(body, loc, value);
      });
  op->getResult(0).replaceAllUsesWith(generic.getResult(0));
  op->erase();
}

class LowerFrontendPass
    : public PassWrapper<LowerFrontendPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerFrontendPass)

  StringRef getArgument() const final { return "lower-frontend"; }
  StringRef getDescription() const final {
    return "Compute the frontend operations no Ball runs in software";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    SmallVector<Operation *> ops;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "frontend")
        ops.push_back(op);
    });
    for (Operation *op : ops) {
      StringRef name = op->getName().getStringRef();
      if (name == "frontend.fp32_to_int8")
        lower(op, true);
      else if (name == "frontend.int8_to_fp32" ||
               name == "frontend.int32_to_fp32") {
        unsigned width = name == "frontend.int8_to_fp32" ? 8 : 32;
        if (!cast<RankedTensorType>(op->getOperand(0).getType())
                 .getElementType()
                 .isInteger(width)) {
          op->emitOpError("requires an INT") << width << " input";
          return signalPassFailure();
        }
        lower(op, false);
      } else {
        op->emitOpError("is not a frontend operation");
        return signalPassFailure();
      }
    }
  }
};
} // namespace

namespace mlir::buddy {
void registerLowerFrontendPass() { PassRegistration<LowerFrontendPass>(); }
} // namespace mlir::buddy

#include "Frontend/FrontendOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include <cstdint>

using namespace mlir;
using namespace ::buddy::frontend;

namespace {
Value integer(OpBuilder &b, Location loc, int64_t value) {
  return arith::ConstantIntOp::create(b, loc, value, 64);
}
Value roundedDivide(OpBuilder &b, Location loc, Value value, Value divisor) {
  Value zero = integer(b, loc, 0), one = integer(b, loc, 1);
  Value negative =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, value, zero);
  Value magnitude = arith::SelectOp::create(
      b, loc, negative, arith::SubIOp::create(b, loc, zero, value), value);
  Value quotient = arith::DivUIOp::create(b, loc, magnitude, divisor);
  Value remainder = arith::RemUIOp::create(b, loc, magnitude, divisor);
  Value twice = arith::ShLIOp::create(b, loc, remainder, one);
  Value odd =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne,
                            arith::AndIOp::create(b, loc, quotient, one), zero);
  Value increment = arith::OrIOp::create(
      b, loc,
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, twice, divisor),
      arith::AndIOp::create(b, loc,
                            arith::CmpIOp::create(b, loc,
                                                  arith::CmpIPredicate::eq,
                                                  twice, divisor),
                            odd));
  Value rounded = arith::AddIOp::create(
      b, loc, quotient,
      arith::ExtUIOp::create(b, loc, b.getI64Type(), increment));
  return arith::SelectOp::create(
      b, loc, negative, arith::SubIOp::create(b, loc, zero, rounded), rounded);
}
Value saturated(OpBuilder &b, Location loc, Value value) {
  Value bounded = arith::MinSIOp::create(
      b, loc, arith::MaxSIOp::create(b, loc, value, integer(b, loc, INT32_MIN)),
      integer(b, loc, INT32_MAX));
  return arith::TruncIOp::create(b, loc, b.getI32Type(), bounded);
}
} // namespace

namespace mlir::buddy {
LogicalResult lowerFixedPoint(Operation *op) {
  OpBuilder b(op);
  Location loc = op->getLoc();
  Value input = op->getOperand(0);
  auto type = cast<RankedTensorType>(input.getType());
  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  auto identity = b.getMultiDimIdentityMap(type.getRank());
  if (isa<FixedMeanOp>(op)) {
    if (!type.hasStaticShape() || type.getRank() != 4 ||
        resultType.getShape() !=
            ArrayRef<int64_t>{type.getDimSize(0), 1, 1, type.getDimSize(3)} ||
        type.getDimSize(1) <= 0 || type.getDimSize(2) <= 0)
      return op->emitOpError(
          "fixed_mean requires static NHWC and a nonempty spatial extent");
    Value output =
        tensor::EmptyOp::create(b, loc, resultType.getShape(), b.getI32Type());
    Value mean =
        linalg::GenericOp::create(
            b, loc, resultType, ValueRange{}, ValueRange{output},
            ArrayRef<AffineMap>{identity},
            SmallVector<utils::IteratorType>(4, utils::IteratorType::parallel),
            [&](OpBuilder &b, Location loc, ValueRange) {
              Value n = linalg::IndexOp::create(b, loc, 0);
              Value channel = linalg::IndexOp::create(b, loc, 3);
              Value zero = arith::ConstantIndexOp::create(b, loc, 0);
              Value one = arith::ConstantIndexOp::create(b, loc, 1);
              Value height =
                  arith::ConstantIndexOp::create(b, loc, type.getDimSize(1));
              Value width =
                  arith::ConstantIndexOp::create(b, loc, type.getDimSize(2));
              // Carry the sum in SSA: a buffer reduction stores it on every
              // pixel when the CPU optimizer cannot prove that the input does
              // not alias.
              auto rows = scf::ForOp::create(
                  b, loc, zero, height, one, ValueRange{integer(b, loc, 0)},
                  [&](OpBuilder &b, Location loc, Value y, ValueRange rowSum) {
                    auto columns = scf::ForOp::create(
                        b, loc, zero, width, one, rowSum,
                        [&](OpBuilder &b, Location loc, Value x,
                            ValueRange sum) {
                          Value value = tensor::ExtractOp::create(
                              b, loc, input, ValueRange{n, y, x, channel});
                          value = arith::ExtSIOp::create(b, loc, b.getI64Type(),
                                                         value);
                          scf::YieldOp::create(
                              b, loc,
                              arith::AddIOp::create(b, loc, sum[0], value)
                                  .getResult());
                        });
                    scf::YieldOp::create(b, loc, columns.getResults());
                  });
              Value value = roundedDivide(
                  b, loc, rows.getResult(0),
                  integer(b, loc, type.getDimSize(1) * type.getDimSize(2)));
              linalg::YieldOp::create(b, loc, saturated(b, loc, value));
            })
            .getResult(0);
    op->getResult(0).replaceAllUsesWith(mean);
    op->erase();
    return success();
  }
  int64_t fraction = cast<IntegerAttr>(op->getAttr("fraction")).getInt();
  if (fraction < 1 || fraction > 24 || type.getRank() < 1 || type != resultType)
    return op->emitOpError(
        "fixed point requires matching I32 tensors and fraction 1..24");
  SmallVector<Value> inputs{input};
  SmallVector<AffineMap> maps{identity};
  auto binary = dyn_cast<FixedBinaryOp>(op);
  auto unary = dyn_cast<FixedUnaryOp>(op);
  auto requant = dyn_cast<FixedRequantOp>(op);
  if (binary) {
    auto rhsType = cast<RankedTensorType>(binary.getRhs().getType());
    if (rhsType.getRank() != type.getRank() ||
        (binary.getKind() != "add" && binary.getKind() != "mul"))
      return op->emitOpError(
          "fixed_binary requires add or mul and equal ranks");
    SmallVector<AffineExpr> dimensions;
    for (int64_t axis = 0; axis < type.getRank(); ++axis) {
      if (rhsType.getDimSize(axis) != 1 &&
          rhsType.getDimSize(axis) != type.getDimSize(axis))
        return op->emitOpError(
            "fixed_binary has incompatible broadcasting dimensions");
      dimensions.push_back(rhsType.getDimSize(axis) == 1
                               ? b.getAffineConstantExpr(0)
                               : b.getAffineDimExpr(axis));
    }
    inputs.push_back(binary.getRhs());
    maps.push_back(
        AffineMap::get(type.getRank(), 0, dimensions, b.getContext()));
  } else if (unary) {
    if (unary.getKind() != "hardswish" && unary.getKind() != "hardsigmoid" &&
        unary.getKind() != "relu")
      return op->emitOpError("unknown fixed unary operation");
  } else {
    auto m = cast<RankedTensorType>(requant.getMultipliers().getType());
    auto s = cast<RankedTensorType>(requant.getShifts().getType());
    if (!m.hasStaticShape() || m.getRank() != 1 || m != s ||
        (m.getDimSize(0) < type.getDimSize(type.getRank() - 1) &&
         m.getDimSize(0) != 16))
      return op->emitOpError("fixed_requant requires one multiplier and shift "
                             "per last-axis channel");
  }
  maps.push_back(identity);
  Value output = tensor::EmptyOp::create(
      b, loc, tensor::getMixedSizes(b, loc, input), b.getI32Type());
  Value result =
      linalg::GenericOp::create(
          b, loc, resultType, inputs, ValueRange{output}, maps,
          SmallVector<utils::IteratorType>(type.getRank(),
                                           utils::IteratorType::parallel),
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value a = arith::ExtSIOp::create(b, loc, b.getI64Type(), args[0]);
            Value value;
            if (requant) {
              Value channel =
                  linalg::IndexOp::create(b, loc, type.getRank() - 1);
              int64_t count =
                  cast<RankedTensorType>(requant.getMultipliers().getType())
                      .getNumElements();
              if (count < type.getShape().back())
                channel = arith::RemUIOp::create(
                    b, loc, channel,
                    arith::ConstantIndexOp::create(b, loc, count));
              Value multiplier = arith::ExtSIOp::create(
                  b, loc, b.getI64Type(),
                  tensor::ExtractOp::create(b, loc, requant.getMultipliers(),
                                            channel));
              Value shift = arith::ExtSIOp::create(
                  b, loc, b.getI64Type(),
                  tensor::ExtractOp::create(b, loc, requant.getShifts(),
                                            channel));
              value = roundedDivide(
                  b, loc, arith::MulIOp::create(b, loc, a, multiplier),
                  arith::ShLIOp::create(b, loc, integer(b, loc, 1), shift));
            } else if (binary) {
              Value rhs =
                  arith::ExtSIOp::create(b, loc, b.getI64Type(), args[1]);
              value = binary.getKind() == "add"
                          ? Value(arith::AddIOp::create(b, loc, a, rhs))
                          : roundedDivide(
                                b, loc, arith::MulIOp::create(b, loc, a, rhs),
                                integer(b, loc, INT64_C(1) << fraction));
            } else if (unary.getKind() == "relu") {
              value = arith::MaxSIOp::create(b, loc, a, integer(b, loc, 0));
            } else {
              Value denominator = integer(b, loc, INT64_C(6) << fraction);
              Value shifted = arith::AddIOp::create(
                  b, loc, a, integer(b, loc, INT64_C(3) << fraction));
              Value clamped = arith::MinSIOp::create(
                  b, loc,
                  arith::MaxSIOp::create(b, loc, shifted, integer(b, loc, 0)),
                  denominator);
              value =
                  unary.getKind() == "hardswish"
                      ? roundedDivide(b, loc,
                                      arith::MulIOp::create(b, loc, a, clamped),
                                      denominator)
                      : roundedDivide(b, loc, clamped, integer(b, loc, 6));
            }
            linalg::YieldOp::create(b, loc, saturated(b, loc, value));
          })
          .getResult(0);
  op->getResult(0).replaceAllUsesWith(result);
  op->erase();
  return success();
}
} // namespace mlir::buddy

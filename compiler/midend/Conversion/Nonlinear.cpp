#include "Frontend/FrontendOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <cstdint>

using namespace mlir;
using namespace ::buddy::frontend;

namespace mlir::buddy {
LogicalResult lowerNonlinear(Operation *op) {
  OpBuilder b(op);
  Location loc = op->getLoc();
  Value input = op->getOperand(0);
  auto type = cast<RankedTensorType>(input.getType());
  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  if (auto softmax = dyn_cast<IntegerSoftmaxOp>(op)) {
    // Along the last dimension, as Frontend.td gives it: Q8 scores, their
    // distances below the row maximum looked up, Q15 normalized by the sum.
    int64_t rank = type.getRank();
    auto identity = b.getMultiDimIdentityMap(rank);
    SmallVector<AffineExpr> rows;
    for (int64_t dim = 0; dim + 1 < rank; ++dim)
      rows.push_back(b.getAffineDimExpr(dim));
    auto rowMap = AffineMap::get(rank, 0, rows, b.getContext());
    SmallVector<utils::IteratorType> parallel(rank,
                                              utils::IteratorType::parallel);
    SmallVector<utils::IteratorType> reduction = parallel;
    reduction.back() = utils::IteratorType::reduction;
    SmallVector<OpFoldResult> sizes = tensor::getMixedSizes(b, loc, input);
    SmallVector<OpFoldResult> rowSizes(sizes.begin(), sizes.end() - 1);
    auto i32 = b.getI32Type();
    auto valuesType = RankedTensorType::get(type.getShape(), i32);
    auto rowType = RankedTensorType::get(type.getShape().drop_back(), i32);
    auto constant = [&](OpBuilder &b, Location loc, int64_t x) -> Value {
      return arith::ConstantIntOp::create(b, loc, x, 32);
    };
    Value fixed =
        linalg::GenericOp::create(
            b, loc, valuesType, ValueRange{input},
            ValueRange{tensor::EmptyOp::create(b, loc, sizes, i32)},
            ArrayRef<AffineMap>{identity, identity}, parallel,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              auto f32 = [&](double x) -> Value {
                return arith::ConstantFloatOp::create(b, loc, b.getF32Type(),
                                                      APFloat(float(x)));
              };
              Value score = arith::MinimumFOp::create(
                  b, loc,
                  arith::MaximumFOp::create(b, loc, args[0], f32(-4096)),
                  f32(4096));
              Value scaled = arith::MulFOp::create(b, loc, score, f32(256));
              Value rounded = math::RoundEvenOp::create(b, loc, scaled);
              linalg::YieldOp::create(
                  b, loc,
                  arith::FPToSIOp::create(b, loc, i32, rounded).getResult());
            })
            .getResult(0);
    Value maximum = tensor::EmptyOp::create(b, loc, rowSizes, i32);
    maximum =
        linalg::FillOp::create(b, loc, constant(b, loc, INT32_MIN), maximum)
            .getResult(0);
    maximum =
        linalg::GenericOp::create(
            b, loc, rowType, ValueRange{fixed}, ValueRange{maximum},
            ArrayRef<AffineMap>{identity, rowMap}, reduction,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              linalg::YieldOp::create(
                  b, loc,
                  arith::MaxSIOp::create(b, loc, args[0], args[1]).getResult());
            })
            .getResult(0);
    Value weights =
        linalg::GenericOp::create(
            b, loc, valuesType, ValueRange{fixed, maximum},
            ValueRange{tensor::EmptyOp::create(b, loc, sizes, i32)},
            ArrayRef<AffineMap>{identity, rowMap, identity}, parallel,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value distance = arith::SubIOp::create(b, loc, args[1], args[0]);
              distance = arith::MinSIOp::create(b, loc, distance,
                                                constant(b, loc, 4095));
              Value index = arith::IndexCastOp::create(b, loc, b.getIndexType(),
                                                       distance);
              linalg::YieldOp::create(
                  b, loc,
                  tensor::ExtractOp::create(b, loc, softmax.getTable(),
                                            ValueRange{index})
                      .getResult());
            })
            .getResult(0);
    Value sum = tensor::EmptyOp::create(b, loc, rowSizes, i32);
    sum = linalg::FillOp::create(b, loc, constant(b, loc, 0), sum).getResult(0);
    sum =
        linalg::GenericOp::create(
            b, loc, rowType, ValueRange{weights}, ValueRange{sum},
            ArrayRef<AffineMap>{identity, rowMap}, reduction,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              linalg::YieldOp::create(
                  b, loc,
                  arith::AddIOp::create(b, loc, args[0], args[1]).getResult());
            })
            .getResult(0);
    Value result =
        linalg::GenericOp::create(
            b, loc, resultType, ValueRange{weights, sum},
            ValueRange{tensor::EmptyOp::create(b, loc, sizes, b.getF32Type())},
            ArrayRef<AffineMap>{identity, rowMap, identity}, parallel,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value scaled = arith::MulIOp::create(b, loc, args[0],
                                                   constant(b, loc, 32768));
              Value half =
                  arith::DivSIOp::create(b, loc, args[1], constant(b, loc, 2));
              Value probability = arith::DivSIOp::create(
                  b, loc, arith::AddIOp::create(b, loc, scaled, half), args[1]);
              Value value =
                  arith::SIToFPOp::create(b, loc, b.getF32Type(), probability);
              Value one = arith::ConstantFloatOp::create(b, loc, b.getF32Type(),
                                                         APFloat(32768.0f));
              linalg::YieldOp::create(
                  b, loc,
                  arith::DivFOp::create(b, loc, value, one).getResult());
            })
            .getResult(0);
    op->getResult(0).replaceAllUsesWith(result);
    op->erase();
    return success();
  }
  if (isa<RowMaxU32Op>(op)) {
    if (!type.hasStaticShape() || type.getRank() != 2 ||
        type.getDimSize(1) <= 0 ||
        resultType.getShape() != ArrayRef<int64_t>{type.getDimSize(0)})
      return op->emitOpError(
          "requires a nonempty static matrix and one result per row");
    Value result =
        tensor::EmptyOp::create(b, loc, resultType.getShape(), b.getI32Type());
    Value zero = arith::ConstantIntOp::create(b, loc, 0, 32);
    result = linalg::FillOp::create(b, loc, zero, result).getResult(0);
    result =
        linalg::GenericOp::create(
            b, loc, resultType, ValueRange{input}, ValueRange{result},
            ArrayRef<AffineMap>{b.getMultiDimIdentityMap(2),
                                AffineMap::get(2, 0, b.getAffineDimExpr(0))},
            ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                          utils::IteratorType::reduction},
            [&](OpBuilder &b, Location loc, ValueRange args) {
              linalg::YieldOp::create(
                  b, loc,
                  arith::MaxUIOp::create(b, loc, args[0], args[1]).getResult());
            })
            .getResult(0);
    op->getResult(0).replaceAllUsesWith(result);
    op->erase();
    return success();
  }
  Value table = op->getOperand(1);
  if (!type.hasStaticShape() || type.getRank() != 2 ||
      type.getShape() != resultType.getShape() ||
      cast<RankedTensorType>(table.getType()).getShape() !=
          ArrayRef<int64_t>{1024})
    return op->emitOpError("requires static matrices and a 1024-entry table");
  auto identity = b.getMultiDimIdentityMap(2);
  auto rowMap = AffineMap::get(2, 0, b.getAffineDimExpr(0));
  SmallVector<utils::IteratorType> parallel(2, utils::IteratorType::parallel);
  SmallVector<utils::IteratorType> reduction{utils::IteratorType::parallel,
                                             utils::IteratorType::reduction};
  if (isa<LookupI32Op>(op)) {
    Value empty =
        tensor::EmptyOp::create(b, loc, type.getShape(), b.getI32Type());
    Value result =
        linalg::GenericOp::create(
            b, loc, resultType, ValueRange{input}, ValueRange{empty},
            ArrayRef<AffineMap>{identity, identity}, parallel,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value low = arith::ConstantIntOp::create(b, loc, -512, 32);
              Value high = arith::ConstantIntOp::create(b, loc, 511, 32);
              Value mask = arith::ConstantIntOp::create(b, loc, 1023, 32);
              Value code = arith::MinSIOp::create(
                  b, loc, arith::MaxSIOp::create(b, loc, args[0], low), high);
              code = arith::AndIOp::create(b, loc, code, mask);
              Value index =
                  arith::IndexCastOp::create(b, loc, b.getIndexType(), code);
              linalg::YieldOp::create(
                  b, loc,
                  tensor::ExtractOp::create(b, loc, table, ValueRange{index})
                      .getResult());
            })
            .getResult(0);
    op->getResult(0).replaceAllUsesWith(result);
    op->erase();
    return success();
  }
  int64_t multiplier = cast<SoftmaxI8Op>(op).getMultiplier();
  if (type.getDimSize(1) <= 0 || multiplier <= 0 || multiplier >= (1LL << 30))
    return op->emitOpError(
        "requires nonempty rows and a positive Q16 multiplier below 2^30");
  auto rowType = RankedTensorType::get({type.getDimSize(0)}, b.getI32Type());
  auto sumType = RankedTensorType::get({type.getDimSize(0)}, b.getI64Type());
  auto valuesType = RankedTensorType::get(type.getShape(), b.getI64Type());
  Value minimum = arith::ConstantIntOp::create(b, loc, INT32_MIN, 32);
  Value maximum =
      tensor::EmptyOp::create(b, loc, rowType.getShape(), b.getI32Type());
  maximum = linalg::FillOp::create(b, loc, minimum, maximum).getResult(0);
  maximum =
      linalg::GenericOp::create(
          b, loc, rowType, ValueRange{input}, ValueRange{maximum},
          ArrayRef<AffineMap>{identity, rowMap}, reduction,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            linalg::YieldOp::create(
                b, loc,
                arith::MaxSIOp::create(b, loc, args[0], args[1]).getResult());
          })
          .getResult(0);
  Value values =
      tensor::EmptyOp::create(b, loc, type.getShape(), b.getI64Type());
  values =
      linalg::GenericOp::create(
          b, loc, valuesType, ValueRange{input, maximum}, ValueRange{values},
          ArrayRef<AffineMap>{identity, rowMap, identity}, parallel,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value x = arith::ExtSIOp::create(b, loc, b.getI64Type(), args[0]);
            Value peak =
                arith::ExtSIOp::create(b, loc, b.getI64Type(), args[1]);
            Value delta = arith::SubIOp::create(b, loc, peak, x);
            delta = arith::MinUIOp::create(
                b, loc, delta, arith::ConstantIntOp::create(b, loc, 1023, 64));
            Value index =
                arith::IndexCastOp::create(b, loc, b.getIndexType(), delta);
            Value entry =
                tensor::ExtractOp::create(b, loc, table, ValueRange{index});
            linalg::YieldOp::create(
                b, loc,
                arith::ExtUIOp::create(b, loc, b.getI64Type(), entry)
                    .getResult());
          })
          .getResult(0);
  Value zero = arith::ConstantIntOp::create(b, loc, 0, 64);
  Value sum =
      tensor::EmptyOp::create(b, loc, sumType.getShape(), b.getI64Type());
  sum = linalg::FillOp::create(b, loc, zero, sum).getResult(0);
  sum = linalg::GenericOp::create(
            b, loc, sumType, ValueRange{values}, ValueRange{sum},
            ArrayRef<AffineMap>{identity, rowMap}, reduction,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              linalg::YieldOp::create(
                  b, loc,
                  arith::AddIOp::create(b, loc, args[0], args[1]).getResult());
            })
            .getResult(0);
  Value reciprocal =
      tensor::EmptyOp::create(b, loc, sumType.getShape(), b.getI64Type());
  auto rowIdentity = b.getMultiDimIdentityMap(1);
  reciprocal =
      linalg::GenericOp::create(
          b, loc, sumType, ValueRange{sum}, ValueRange{reciprocal},
          ArrayRef<AffineMap>{rowIdentity, rowIdentity},
          ArrayRef<utils::IteratorType>{utils::IteratorType::parallel},
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value numerator =
                arith::ConstantIntOp::create(b, loc, multiplier << 32, 64);
            linalg::YieldOp::create(
                b, loc,
                arith::DivUIOp::create(b, loc, numerator, args[0]).getResult());
          })
          .getResult(0);
  Value output =
      tensor::EmptyOp::create(b, loc, type.getShape(), b.getI8Type());
  Value result =
      linalg::GenericOp::create(
          b, loc, resultType, ValueRange{values, reciprocal},
          ValueRange{output}, ArrayRef<AffineMap>{identity, rowMap, identity},
          parallel,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            auto constant = [&](int64_t x) -> Value {
              return arith::ConstantIntOp::create(b, loc, x, 64);
            };
            Value product = arith::MulIOp::create(b, loc, args[0], args[1]);
            Value quotient =
                arith::ShRUIOp::create(b, loc, product, constant(48));
            Value remainder = arith::AndIOp::create(b, loc, product,
                                                    constant((1LL << 48) - 1));
            Value odd = arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate::ne,
                arith::AndIOp::create(b, loc, quotient, constant(1)),
                constant(0));
            Value increment = arith::OrIOp::create(
                b, loc,
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt,
                                      remainder, constant(1LL << 47)),
                arith::AndIOp::create(
                    b, loc, odd,
                    arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                          remainder, constant(1LL << 47))));
            Value rounded = arith::AddIOp::create(
                b, loc, quotient,
                arith::ExtUIOp::create(b, loc, b.getI64Type(), increment));
            rounded = arith::MinUIOp::create(b, loc, rounded, constant(127));
            linalg::YieldOp::create(
                b, loc,
                arith::TruncIOp::create(b, loc, b.getI8Type(), rounded)
                    .getResult());
          })
          .getResult(0);
  op->getResult(0).replaceAllUsesWith(result);
  op->erase();
  return success();
}
} // namespace mlir::buddy

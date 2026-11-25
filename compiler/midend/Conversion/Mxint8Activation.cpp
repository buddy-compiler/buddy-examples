#include "Frontend/FrontendOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace ::buddy::frontend;

namespace {
Value integer(OpBuilder &b, Location loc, int64_t value) {
  return arith::ConstantIntOp::create(b, loc, value, 64);
}

Value roundShift(OpBuilder &b, Location loc, Value value, Value amount) {
  auto c = [&](int64_t n) { return integer(b, loc, n); };
  Value negative =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, value, c(0));
  Value magnitude = arith::SelectOp::create(
      b, loc, negative, arith::SubIOp::create(b, loc, c(0), value), value);
  Value shift = arith::MinSIOp::create(
      b, loc, arith::MaxSIOp::create(b, loc, amount, c(0)), c(63));
  Value quotient = arith::ShRUIOp::create(b, loc, magnitude, shift);
  Value mask = arith::SubIOp::create(
      b, loc, arith::ShLIOp::create(b, loc, c(1), shift), c(1));
  Value remainder = arith::AndIOp::create(b, loc, magnitude, mask);
  Value halfShift = arith::MaxSIOp::create(
      b, loc, arith::SubIOp::create(b, loc, shift, c(1)), c(0));
  Value half = arith::ShLIOp::create(b, loc, c(1), halfShift);
  Value odd = arith::CmpIOp::create(
      b, loc, arith::CmpIPredicate::ne,
      arith::AndIOp::create(b, loc, quotient, c(1)), c(0));
  Value greater =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, remainder, half);
  Value equal =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, remainder, half);
  Value increment = arith::AndIOp::create(
      b, loc,
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt, amount, c(0)),
      arith::OrIOp::create(b, loc, greater,
                           arith::AndIOp::create(b, loc, equal, odd)));
  Value rounded = arith::AddIOp::create(
      b, loc, quotient,
      arith::ExtUIOp::create(b, loc, b.getI64Type(), increment));
  rounded = arith::SelectOp::create(
      b, loc, negative, arith::SubIOp::create(b, loc, c(0), rounded), rounded);
  rounded = arith::SelectOp::create(
      b, loc,
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, amount, c(64)),
      c(0), rounded);
  Value left = arith::MinSIOp::create(
      b, loc,
      arith::MaxSIOp::create(b, loc,
                             arith::SubIOp::create(b, loc, c(0), amount), c(0)),
      c(14));
  return arith::SelectOp::create(
      b, loc,
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, amount, c(0)),
      arith::ShLIOp::create(b, loc, value, left), rounded);
}
} // namespace

namespace mlir::buddy {
LogicalResult lowerMxint8Activation(Operation *operation) {
  auto op = cast<MXINT8ActivationOp>(operation);
  auto type = cast<RankedTensorType>(op.getInput().getType());
  auto resultType = cast<RankedTensorType>(op.getResult().getType());
  auto tableType = cast<RankedTensorType>(op.getTable().getType());
  auto gateType = cast<RankedTensorType>(op.getGate().getType());
  int64_t rank = type.getRank();
  if (rank < 2 || op.getAxis() != rank - 1 || type.isDynamicDim(rank - 1) ||
      resultType.getRank() != rank || resultType.isDynamicDim(rank - 1) ||
      tableType.getShape() != ArrayRef<int64_t>{4096} || gateType != type)
    return op.emitOpError(
        "requires packed last-axis rows and a 4096-entry Q26 table");
  int64_t packed = resultType.getShape().back(), k = packed * 32 / 33;
  int64_t inputK = k;
  if (k <= 0 || k % 32 || packed != k + k / 32 ||
      type.getShape().back() != inputK + inputK / 32)
    return op.emitOpError("has inconsistent packed MXINT8 dimensions");
  OpBuilder b(op);
  Location loc = op.getLoc();
  int64_t staticRows = 1;
  ReassociationIndices leading;
  for (int64_t axis = 0; axis < rank - 1; ++axis) {
    leading.push_back(axis);
    if (type.isDynamicDim(axis))
      staticRows = ShapedType::kDynamic;
    else if (staticRows != ShapedType::kDynamic)
      staticRows *= type.getDimSize(axis);
  }
  SmallVector<ReassociationIndices> groups{leading, {rank - 1}};
  Value input = op.getInput(), gateInput = op.getGate();
  if (rank > 2) {
    auto flatInputType = RankedTensorType::get(
        {staticRows, inputK + inputK / 32}, b.getI8Type());
    input =
        tensor::CollapseShapeOp::create(b, loc, flatInputType, input, groups);
    gateInput = tensor::CollapseShapeOp::create(b, loc, flatInputType,
                                                gateInput, groups);
  }
  SmallVector<Value> dynamic;
  if (staticRows == ShapedType::kDynamic)
    dynamic.push_back(tensor::DimOp::create(b, loc, input, 0));
  int64_t blocks = k / 32;
  auto valuesType =
      RankedTensorType::get({staticRows, blocks, 32}, b.getI64Type());
  auto maximumType =
      RankedTensorType::get({staticRows, blocks}, b.getI64Type());
  auto exponentType =
      RankedTensorType::get({staticRows, blocks}, b.getI64Type());
  auto flatType = RankedTensorType::get({staticRows, packed}, b.getI8Type());
  auto exponent = [&](OpBuilder &b, Value row, Value block,
                      bool gate) -> Value {
    Value at = arith::AddIOp::create(
        b, loc, block, arith::ConstantIndexOp::create(b, loc, inputK));
    Value code = tensor::ExtractOp::create(b, loc, gate ? gateInput : input,
                                           ValueRange{row, at});
    return arith::SubIOp::create(
        b, loc, arith::ExtUIOp::create(b, loc, b.getI64Type(), code),
        integer(b, loc, 127));
  };
  Value empty = tensor::EmptyOp::create(b, loc, valuesType.getShape(),
                                        b.getI64Type(), dynamic);
  Value values =
      linalg::GenericOp::create(
          b, loc, valuesType, ValueRange{}, ValueRange{empty},
          ArrayRef<AffineMap>{b.getMultiDimIdentityMap(3)},
          SmallVector<utils::IteratorType>(3, utils::IteratorType::parallel),
          [&](OpBuilder &b, Location loc, ValueRange) {
            auto c = [&](int64_t n) { return integer(b, loc, n); };
            Value row = linalg::IndexOp::create(b, loc, 0),
                  block = linalg::IndexOp::create(b, loc, 1);
            Value lane = linalg::IndexOp::create(b, loc, 2);
            Value at = arith::AddIOp::create(
                b, loc,
                arith::MulIOp::create(
                    b, loc, block, arith::ConstantIndexOp::create(b, loc, 32)),
                lane);
            Value code = arith::ExtSIOp::create(
                b, loc, b.getI64Type(),
                tensor::ExtractOp::create(b, loc, input, ValueRange{row, at}));
            Value e = exponent(b, row, block, false);
            Value bounded = arith::MinSIOp::create(
                b, loc, arith::MaxSIOp::create(b, loc, e, c(-24)), c(4));
            Value x = arith::ShLIOp::create(
                b, loc, code, arith::AddIOp::create(b, loc, bounded, c(26)));
            Value position = arith::AddIOp::create(b, loc, x, c(16LL << 26));
            Value index = arith::MinSIOp::create(
                b, loc,
                arith::MaxSIOp::create(
                    b, loc, arith::ShRSIOp::create(b, loc, position, c(19)),
                    c(0)),
                c(4095));
            Value next = arith::MinSIOp::create(
                b, loc, arith::AddIOp::create(b, loc, index, c(1)), c(4095));
            Value first = arith::ExtSIOp::create(
                b, loc, b.getI64Type(),
                tensor::ExtractOp::create(
                    b, loc, op.getTable(),
                    ValueRange{arith::IndexCastOp::create(
                        b, loc, b.getIndexType(), index)}));
            Value last = arith::ExtSIOp::create(
                b, loc, b.getI64Type(),
                tensor::ExtractOp::create(
                    b, loc, op.getTable(),
                    ValueRange{arith::IndexCastOp::create(
                        b, loc, b.getIndexType(), next)}));
            last = arith::SelectOp::create(
                b, loc,
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, index,
                                      c(4095)),
                c(16LL << 26), last);
            Value fraction =
                arith::AndIOp::create(b, loc, position, c((1 << 19) - 1));
            Value delta = arith::MulIOp::create(
                b, loc, arith::SubIOp::create(b, loc, last, first), fraction);
            Value value =
                roundShift(b, loc,
                           arith::AddIOp::create(
                               b, loc, first, roundShift(b, loc, delta, c(19))),
                           arith::AddIOp::create(b, loc, e, c(10)));
            value = arith::SelectOp::create(
                b, loc,
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sle, x,
                                      c(-(16LL << 26))),
                c(0), value);
            value = arith::SelectOp::create(
                b, loc,
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, x,
                                      c(16LL << 26)),
                arith::ShLIOp::create(b, loc, code, c(16)), value);
            value = arith::SelectOp::create(
                b, loc,
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, e,
                                      c(-24)),
                arith::ShLIOp::create(b, loc, code, c(15)), value);
            if (op.getGated()) {
              Value gate = arith::ExtSIOp::create(
                  b, loc, b.getI64Type(),
                  tensor::ExtractOp::create(b, loc, gateInput,
                                            ValueRange{row, at}));
              value = arith::MulIOp::create(b, loc, value, gate);
            }
            linalg::YieldOp::create(b, loc, value);
          })
          .getResult(0);
  empty = tensor::EmptyOp::create(b, loc, maximumType.getShape(),
                                  b.getI64Type(), dynamic);
  Value initial =
      linalg::FillOp::create(b, loc, integer(b, loc, 0), empty).getResult(0);
  auto blockMap = AffineMap::get(
      3, 0, {b.getAffineDimExpr(0), b.getAffineDimExpr(1)}, b.getContext());
  Value maximum =
      linalg::GenericOp::create(
          b, loc, maximumType, ValueRange{values}, ValueRange{initial},
          ArrayRef<AffineMap>{b.getMultiDimIdentityMap(3), blockMap},
          ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                        utils::IteratorType::parallel,
                                        utils::IteratorType::reduction},
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value magnitude = arith::SelectOp::create(
                b, loc,
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
                                      args[0], integer(b, loc, 0)),
                arith::SubIOp::create(b, loc, integer(b, loc, 0), args[0]),
                args[0]);
            linalg::YieldOp::create(
                b, loc,
                arith::MaxUIOp::create(b, loc, magnitude, args[1]).getResult());
          })
          .getResult(0);
  empty = tensor::EmptyOp::create(b, loc, exponentType.getShape(),
                                  b.getI64Type(), dynamic);
  Value exponents =
      linalg::GenericOp::create(
          b, loc, exponentType, ValueRange{maximum}, ValueRange{empty},
          ArrayRef<AffineMap>{b.getMultiDimIdentityMap(2),
                              b.getMultiDimIdentityMap(2)},
          SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
          [&](OpBuilder &b, Location loc, ValueRange args) {
            auto c = [&](int64_t n) { return integer(b, loc, n); };
            Value row = linalg::IndexOp::create(b, loc, 0),
                  block = linalg::IndexOp::create(b, loc, 1);
            Value scale = exponent(b, row, block, false);
            if (op.getGated())
              scale = arith::AddIOp::create(b, loc, scale,
                                            exponent(b, row, block, true));
            Value highest = arith::SubIOp::create(
                b, loc, c(63),
                math::CountLeadingZerosOp::create(b, loc, args[0]));
            Value result = arith::SubIOp::create(
                b, loc, arith::AddIOp::create(b, loc, scale, highest), c(22));
            result = arith::MinSIOp::create(
                b, loc, arith::MaxSIOp::create(b, loc, result, c(-127)),
                c(127));
            result = arith::SelectOp::create(
                b, loc,
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, args[0],
                                      c(0)),
                c(0), result);
            linalg::YieldOp::create(b, loc, result);
          })
          .getResult(0);
  empty = tensor::EmptyOp::create(b, loc, flatType.getShape(), b.getI8Type(),
                                  dynamic);
  Value result =
      linalg::GenericOp::create(
          b, loc, flatType, ValueRange{}, ValueRange{empty},
          ArrayRef<AffineMap>{b.getMultiDimIdentityMap(2)},
          SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
          [&](OpBuilder &b, Location loc, ValueRange) {
            auto c = [&](int64_t n) { return integer(b, loc, n); };
            Value row = linalg::IndexOp::create(b, loc, 0),
                  column = linalg::IndexOp::create(b, loc, 1);
            Value boundary = arith::ConstantIndexOp::create(b, loc, k);
            Value at = arith::MinUIOp::create(
                b, loc, column, arith::ConstantIndexOp::create(b, loc, k - 1));
            Value block = arith::DivUIOp::create(
                b, loc, at, arith::ConstantIndexOp::create(b, loc, 32));
            Value lane = arith::RemUIOp::create(
                b, loc, at, arith::ConstantIndexOp::create(b, loc, 32));
            Value value = tensor::ExtractOp::create(
                b, loc, values, ValueRange{row, block, lane});
            Value e = tensor::ExtractOp::create(b, loc, exponents,
                                                ValueRange{row, block});
            Value scale = exponent(b, row, block, false);
            if (op.getGated())
              scale = arith::AddIOp::create(b, loc, scale,
                                            exponent(b, row, block, true));
            Value shift = arith::AddIOp::create(
                b, loc, arith::SubIOp::create(b, loc, e, scale), c(16));
            Value sign = arith::SelectOp::create(
                b, loc,
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, value,
                                      c(0)),
                c(-127), c(127));
            sign = arith::SelectOp::create(
                b, loc,
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, value,
                                      c(0)),
                c(0), sign);
            Value code = arith::SelectOp::create(
                b, loc,
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, shift,
                                      c(-6)),
                sign, roundShift(b, loc, value, shift));
            code = arith::MinSIOp::create(
                b, loc, arith::MaxSIOp::create(b, loc, code, c(-127)), c(127));
            Value scaleBlock = arith::MaxSIOp::create(
                b, loc, arith::SubIOp::create(b, loc, column, boundary),
                arith::ConstantIndexOp::create(b, loc, 0));
            scaleBlock = arith::MinUIOp::create(
                b, loc, scaleBlock,
                arith::ConstantIndexOp::create(b, loc, blocks - 1));
            Value stored = arith::AddIOp::create(
                b, loc,
                tensor::ExtractOp::create(b, loc, exponents,
                                          ValueRange{row, scaleBlock}),
                c(127));
            Value output = arith::SelectOp::create(
                b, loc,
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, column,
                                      boundary),
                code, stored);
            linalg::YieldOp::create(
                b, loc,
                arith::TruncIOp::create(b, loc, b.getI8Type(), output)
                    .getResult());
          })
          .getResult(0);
  if (rank > 2) {
    SmallVector<OpFoldResult> sizes;
    for (int64_t axis = 0; axis < rank - 1; ++axis)
      sizes.push_back(
          type.isDynamicDim(axis)
              ? OpFoldResult(tensor::DimOp::create(b, loc, op.getInput(), axis)
                                 .getResult())
              : OpFoldResult(b.getIndexAttr(type.getDimSize(axis))));
    sizes.push_back(b.getIndexAttr(packed));
    result = tensor::ExpandShapeOp::create(b, loc, resultType, result, groups,
                                           sizes);
  }
  op.getResult().replaceAllUsesWith(result);
  op.erase();
  return success();
}
} // namespace mlir::buddy

#include "Frontend/FrontendOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace ::buddy::frontend;

namespace mlir::buddy {
LogicalResult lowerConcatColumns(ConcatColumnsOp op) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  auto type = cast<RankedTensorType>(op.getResult().getType());
  if (!type.hasStaticShape() || type.getRank() != 2 || op.getInputs().empty())
    return op.emitOpError("requires static matrices");
  Value result =
      tensor::EmptyOp::create(b, loc, type.getShape(), type.getElementType());
  int64_t column = 0;
  for (Value input : op.getInputs()) {
    auto source = cast<RankedTensorType>(input.getType());
    if (!source.hasStaticShape() || source.getRank() != 2 ||
        source.getDimSize(0) != type.getDimSize(0) ||
        source.getElementType() != type.getElementType())
      return op.emitOpError("requires matching matrix rows and element types");
    result = tensor::InsertSliceOp::create(
        b, loc, input, result,
        ArrayRef<OpFoldResult>{b.getIndexAttr(0), b.getIndexAttr(column)},
        ArrayRef<OpFoldResult>{b.getIndexAttr(source.getDimSize(0)),
                               b.getIndexAttr(source.getDimSize(1))},
        SmallVector<OpFoldResult>(2, b.getIndexAttr(1)));
    column += source.getDimSize(1);
  }
  if (column != type.getDimSize(1))
    return op.emitOpError("result columns must equal the input column sum");
  op.getResult().replaceAllUsesWith(result);
  op.erase();
  return success();
}

LogicalResult lowerGatherRows(GatherRowsOp op) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  auto resultType = cast<RankedTensorType>(op.getResult().getType());
  auto indicesType = cast<RankedTensorType>(op.getIndices().getType());
  if (!resultType.hasStaticShape() || resultType.getRank() != 2 ||
      indicesType.getShape() != ArrayRef<int64_t>{resultType.getDimSize(0)} ||
      op.getInputs().empty())
    return op.emitOpError(
        "requires row indices, matrices and a matching result matrix");
  auto index = [&](int64_t x) -> Value {
    return arith::ConstantIndexOp::create(b, loc, x);
  };
  Value zero = index(0), one = index(1);
  Type element = resultType.getElementType();
  int64_t columns = resultType.getDimSize(1), end = 0;
  auto layout =
      StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic,
                             {ShapedType::kDynamic, ShapedType::kDynamic});
  auto matrixType =
      MemRefType::get({ShapedType::kDynamic, columns}, element, layout);
  SmallVector<Value> matrices;
  SmallVector<int64_t> starts;
  for (Value tensor : op.getInputs()) {
    auto type = cast<RankedTensorType>(tensor.getType());
    if (!type.hasStaticShape() || type.getRank() != 2 ||
        type.getDimSize(1) != columns || type.getElementType() != element)
      return op.emitOpError(
          "requires matrices with the same columns and element type");
    Value memory = bufferization::ToBufferOp::create(
        b, loc, MemRefType::get(type.getShape(), element, layout), tensor,
        true);
    matrices.push_back(memref::CastOp::create(b, loc, matrixType, memory));
    starts.push_back(end);
    end += type.getDimSize(0);
  }
  Value indices = bufferization::ToBufferOp::create(
      b, loc, MemRefType::get(indicesType.getShape(), b.getI64Type()),
      op.getIndices(), true);
  Value output = memref::AllocOp::create(
      b, loc, MemRefType::get(resultType.getShape(), element));
  auto rows =
      scf::ForOp::create(b, loc, zero, index(resultType.getDimSize(0)), one);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(rows.getBody());
    Value row = rows.getInductionVar();
    Value at = memref::LoadOp::create(b, loc, indices, row);
    at = arith::IndexCastOp::create(b, loc, b.getIndexType(), at);
    Value source = matrices.back();
    Value local = arith::SubIOp::create(b, loc, at, index(starts.back()));
    for (int64_t i = matrices.size() - 2; i >= 0; --i) {
      Value before = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                           at, index(starts[i + 1]));
      auto selected = scf::IfOp::create(
          b, loc, TypeRange{matrixType, b.getIndexType()}, before, true);
      {
        OpBuilder::InsertionGuard branch(b);
        b.setInsertionPointToStart(&selected.getThenRegion().front());
        scf::YieldOp::create(
            b, loc,
            ValueRange{matrices[i],
                       arith::SubIOp::create(b, loc, at, index(starts[i]))});
        b.setInsertionPointToStart(&selected.getElseRegion().front());
        scf::YieldOp::create(b, loc, ValueRange{source, local});
      }
      source = selected.getResult(0);
      local = selected.getResult(1);
    }
    auto cols = scf::ForOp::create(b, loc, zero, index(columns), one);
    b.setInsertionPointToStart(cols.getBody());
    Value column = cols.getInductionVar();
    Value value =
        memref::LoadOp::create(b, loc, source, ValueRange{local, column});
    memref::StoreOp::create(b, loc, value, output, ValueRange{row, column});
  }
  Value result =
      bufferization::ToTensorOp::create(b, loc, resultType, output, true, true);
  op.getResult().replaceAllUsesWith(result);
  op.erase();
  return success();
}
} // namespace mlir::buddy

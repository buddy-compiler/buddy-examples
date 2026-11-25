//===- FrontendDialect.cpp - Frontend dialect -----------------------------===//

#include "Frontend/FrontendDialect.h"
#include "Frontend/FrontendOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace ::buddy::frontend;

#include "FrontendDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Frontend.cpp.inc"

namespace {
struct Inliner : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Operation *op, Region *, bool, IRMapping &) const final {
    return isMemoryEffectFree(op);
  }
};
} // namespace

void FrontendDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Frontend.cpp.inc"
      >();
  addInterfaces<Inliner>();
}

LogicalResult IntegerSoftmaxOp::verify() {
  auto input = getInput().getType();
  if (input.getRank() < 1)
    return emitOpError("requires a score row");
  int64_t width = input.getShape().back();
  if (!ShapedType::isDynamic(width) && (width < 1 || width > 128))
    return emitOpError("requires between one and 128 input columns");
  if (getResult().getType() != input)
    return emitOpError("requires matching input and output types");
  if (getTable().getType().getShape() != ArrayRef<int64_t>{4096})
    return emitOpError("requires an I32[4096] exponential table");
  return success();
}

LogicalResult
FP32ToMXINT8Op::reifyResultShapes(OpBuilder &builder,
                                  ReifiedRankedShapedTypeDims &shapes) {
  auto sizes = tensor::getMixedSizes(builder, getLoc(), getInput());
  Value length =
      builder.createOrFold<tensor::DimOp>(getLoc(), getInput(), getAxis());
  Value block = arith::ConstantIndexOp::create(builder, getLoc(), 32);
  Value packed = arith::ConstantIndexOp::create(builder, getLoc(), 33);
  Value count =
      builder.createOrFold<arith::CeilDivUIOp>(getLoc(), length, block);
  sizes[getAxis()] =
      builder.createOrFold<arith::MulIOp>(getLoc(), count, packed);
  shapes.push_back(std::move(sizes));
  return success();
}

LogicalResult
MXINT8ToFP32Op::reifyResultShapes(OpBuilder &builder,
                                  ReifiedRankedShapedTypeDims &shapes) {
  auto sizes = tensor::getMixedSizes(builder, getLoc(), getInput());
  Value length =
      builder.createOrFold<tensor::DimOp>(getLoc(), getInput(), getAxis());
  Value packed = arith::ConstantIndexOp::create(builder, getLoc(), 33);
  Value block = arith::ConstantIndexOp::create(builder, getLoc(), 32);
  Value count = builder.createOrFold<arith::DivUIOp>(getLoc(), length, packed);
  sizes[getAxis()] =
      builder.createOrFold<arith::MulIOp>(getLoc(), count, block);
  shapes.push_back(std::move(sizes));
  return success();
}

LogicalResult
MXFP4ToFP32Op::reifyResultShapes(OpBuilder &builder,
                                 ReifiedRankedShapedTypeDims &shapes) {
  // K values in K / 2 + K / 32 bytes: 17 bytes a block of 32.
  auto sizes = tensor::getMixedSizes(builder, getLoc(), getInput());
  Value length =
      builder.createOrFold<tensor::DimOp>(getLoc(), getInput(), getAxis());
  Value packed = arith::ConstantIndexOp::create(builder, getLoc(), 17);
  Value block = arith::ConstantIndexOp::create(builder, getLoc(), 32);
  Value count = builder.createOrFold<arith::DivUIOp>(getLoc(), length, packed);
  sizes[getAxis()] =
      builder.createOrFold<arith::MulIOp>(getLoc(), count, block);
  shapes.push_back(std::move(sizes));
  return success();
}

LogicalResult
MXINT5ToFP32Op::reifyResultShapes(OpBuilder &builder,
                                  ReifiedRankedShapedTypeDims &shapes) {
  auto sizes = tensor::getMixedSizes(builder, getLoc(), getInput());
  Value length =
      builder.createOrFold<tensor::DimOp>(getLoc(), getInput(), getAxis());
  Value packed = arith::ConstantIndexOp::create(builder, getLoc(), 22);
  Value block = arith::ConstantIndexOp::create(builder, getLoc(), 32);
  Value count = builder.createOrFold<arith::DivUIOp>(getLoc(), length, packed);
  sizes[getAxis()] =
      builder.createOrFold<arith::MulIOp>(getLoc(), count, block);
  shapes.push_back(std::move(sizes));
  return success();
}

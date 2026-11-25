//===- LowerFrontendFakeOp.cpp - The frontend fake operations in software -===//
//
// The frontend inserts the operations of the frontend dialect; a design's own
// passes run the ones its hardware runs. This pass, run after them, computes
// every remaining one in software.
//
// The quantization computes on tensors exactly as its reference in
// compiler/frontend/quant.py:
//
//   fp32_to_int8(x, multiplier) = clamp(round_half_even(x * multiplier),
//                                       -128, 127)
//   int8_to_fp32(q, step)      = q * step
//   int32_to_fp32(q, step)     = q * step
//
// The scale broadcasts along its dimensions of size one.
//
// MXINT8 splits `axis` (length K) into blocks of 32 values sharing one
// exponent e = floor(log2(max |x|)) - 6, at least -127, and 0 for an all-zero
// block. The packed tensor holds along `axis` the K codes
// clamp(round_half_even(x * 2^-e), -127, 127), then the K / 32 bytes e + 127;
// a value is code * 2^e. Every power of two is formed from its exponent bits,
// so the scaling is exact.
//
// Every size may be dynamic (the token count of a prefill): every tensor
// takes them from the input with tensor.dim. A dynamic K is zero-padded to
// whole blocks, Kp = ceil(K / 32) * 32: a zero changes neither the maximum of
// its block nor any product, so the packed tensor holds Kp codes and Kp / 32
// exponents, and its values are the Kp values, zeros after K.
//
// A trace point becomes calls of the trace runtime (stack/runtime/trace): a
// cycle trace calls buddyTraceCycleStartPath at its start and
// buddyTraceCycleEndPath at its end, a tensor trace passes the traced tensor,
// flattened, to buddyTraceTensor<type>Path at its end. Each call takes the id
// and the id path (its depth, then four components, -1 past the depth). The
// tensor is read through bufferization.to_buffer: bufferization gives it its
// buffer, copied to an identity layout when its own layout is another. With
// gpu=true (the values of a GPU target live in device memory) a tensor trace
// is an error.
//
//===----------------------------------------------------------------------===//

#include "Frontend/FrontendOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::buddy {
LogicalResult lowerFixedPoint(Operation *op);
LogicalResult lowerNonlinear(Operation *op);
LogicalResult lowerMxint8Activation(Operation *op);
LogicalResult lowerGatherRows(::buddy::frontend::GatherRowsOp op);
LogicalResult lowerConcatColumns(::buddy::frontend::ConcatColumnsOp op);
} // namespace mlir::buddy

using namespace mlir;
using namespace ::buddy::frontend;

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

void lowerLookup(LookupI8Op op) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  auto type = cast<RankedTensorType>(op.getResult().getType());
  auto identity =
      AffineMap::getMultiDimIdentityMap(type.getRank(), b.getContext());
  Value empty = tensor::EmptyOp::create(
      b, loc, tensor::getMixedSizes(b, loc, op.getInput()), b.getI8Type());
  auto result = linalg::GenericOp::create(
      b, loc, type, ValueRange{op.getInput()}, ValueRange{empty},
      ArrayRef<AffineMap>{identity, identity},
      SmallVector<utils::IteratorType>(type.getRank(),
                                       utils::IteratorType::parallel),
      [&](OpBuilder &body, Location loc, ValueRange args) {
        Value code =
            arith::ExtUIOp::create(body, loc, body.getI32Type(), args[0]);
        Value index =
            arith::IndexCastOp::create(body, loc, body.getIndexType(), code);
        if (cast<RankedTensorType>(op.getTable().getType()).getNumElements() !=
            256) {
          Value channel =
              linalg::IndexOp::create(body, loc, type.getRank() - 1);
          Value offset = arith::MulIOp::create(
              body, loc, channel,
              arith::ConstantIndexOp::create(body, loc, 256));
          index = arith::AddIOp::create(body, loc, index, offset);
        }
        Value value = tensor::ExtractOp::create(body, loc, op.getTable(),
                                                ValueRange{index});
        linalg::YieldOp::create(body, loc, value);
      });
  op.getResult().replaceAllUsesWith(result.getResult(0));
  op.erase();
}

LogicalResult lowerScaledAdd(ScaledAddI8Op op) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  auto type = cast<RankedTensorType>(op.getResult().getType());
  if (op.getLhs().getType() != type || op.getRhs().getType() != type ||
      !op.getLhsRatio().isFinite() || !op.getRhsRatio().isFinite() ||
      op.getLhsRatio().isNegative() || op.getLhsRatio().isZero() ||
      op.getRhsRatio().isNegative() || op.getRhsRatio().isZero())
    return op.emitOpError(
        "requires matching shapes and finite positive ratios");
  Value init = tensor::EmptyOp::create(b, loc, type.getShape(), b.getI8Type());
  SmallVector<AffineMap> maps(3, b.getMultiDimIdentityMap(type.getRank()));
  auto result = linalg::GenericOp::create(
      b, loc, type, ValueRange{op.getLhs(), op.getRhs()}, ValueRange{init},
      maps,
      SmallVector<utils::IteratorType>(type.getRank(),
                                       utils::IteratorType::parallel),
      [&](OpBuilder &body, Location loc, ValueRange args) {
        Value lhs =
            arith::SIToFPOp::create(body, loc, body.getF32Type(), args[0]);
        Value rhs =
            arith::SIToFPOp::create(body, loc, body.getF32Type(), args[1]);
        lhs = arith::MulFOp::create(
            body, loc, lhs,
            arith::ConstantOp::create(body, loc, op.getLhsRatioAttr()));
        rhs = arith::MulFOp::create(
            body, loc, rhs,
            arith::ConstantOp::create(body, loc, op.getRhsRatioAttr()));
        Value sum = arith::AddFOp::create(body, loc, lhs, rhs);
        sum = math::RoundEvenOp::create(body, loc, sum);
        sum = arith::MaximumFOp::create(
            body, loc, sum,
            arith::ConstantOp::create(body, loc, body.getF32FloatAttr(-128)));
        sum = arith::MinimumFOp::create(
            body, loc, sum,
            arith::ConstantOp::create(body, loc, body.getF32FloatAttr(127)));
        linalg::YieldOp::create(
            body, loc,
            arith::FPToSIOp::create(body, loc, body.getI8Type(), sum)
                .getResult());
      });
  op.getResult().replaceAllUsesWith(result.getResult(0));
  op.erase();
  return success();
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
  Value init = tensor::EmptyOp::create(
      builder, loc, tensor::getMixedSizes(builder, loc, input),
      result.getElementType());
  auto generic = linalg::GenericOp::create(
      builder, loc, result, ValueRange{input, scale}, ValueRange{init}, maps,
      iterators, [&](OpBuilder &body, Location loc, ValueRange args) {
        Value value;
        if (quantize) {
          Value source = args[0];
          if (source.getType().isInteger(32))
            source =
                arith::SIToFPOp::create(body, loc, body.getF32Type(), source);
          Value scaled = arith::MulFOp::create(body, loc, source, args[1]);
          Value rounded = math::RoundEvenOp::create(body, loc, scaled);
          Value low =
              arith::ConstantOp::create(body, loc, body.getF32FloatAttr(-128));
          Value high =
              arith::ConstantOp::create(body, loc, body.getF32FloatAttr(127));
          if (result.getElementType().isInteger(32)) {
            rounded =
                arith::ExtFOp::create(body, loc, body.getF64Type(), rounded);
            low = arith::ConstantOp::create(
                body, loc, body.getF64FloatAttr(-2147483648.0));
            high = arith::ConstantOp::create(
                body, loc, body.getF64FloatAttr(2147483647.0));
          }
          Value clamped = arith::MinimumFOp::create(
              body, loc, arith::MaximumFOp::create(body, loc, rounded, low),
              high);
          value = arith::FPToSIOp::create(body, loc, result.getElementType(),
                                          clamped);
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

// The reassociation that splits `axis` into (blocks, 32).
SmallVector<ReassociationIndices> blockGroups(int64_t rank, int64_t axis) {
  SmallVector<ReassociationIndices> groups;
  for (int64_t dim = 0, expanded = 0; dim < rank; ++dim) {
    if (dim == axis) {
      groups.push_back({expanded, expanded + 1});
      expanded += 2;
    } else {
      groups.push_back({expanded++});
    }
  }
  return groups;
}

template <typename T>
SmallVector<T> withAxis(ArrayRef<T> shape, int64_t axis, ArrayRef<T> sizes) {
  SmallVector<T> result(shape.begin(), shape.begin() + axis);
  result.append(sizes.begin(), sizes.end());
  result.append(shape.begin() + axis + 1, shape.end());
  return result;
}

// Maps of a generic over the blocked shape (rank + 1 dimensions, the 32
// values of a block at axis + 1): every value, and the block of a value.
AffineMap blockedMap(MLIRContext *context, int64_t rank) {
  return AffineMap::getMultiDimIdentityMap(rank + 1, context);
}

AffineMap blockMap(MLIRContext *context, int64_t rank, int64_t axis) {
  SmallVector<AffineExpr> results;
  for (int64_t dim = 0; dim <= rank; ++dim)
    if (dim != axis + 1)
      results.push_back(getAffineDimExpr(dim, context));
  return AffineMap::get(rank + 1, 0, results, context);
}

// 2^(exponent - 127) for an i32 biased exponent in [1, 254].
Value power(OpBuilder &body, Location loc, Value biased) {
  Value bits = arith::ShLIOp::create(
      body, loc, biased,
      arith::ConstantIntOp::create(body, loc, body.getI32Type(), 23));
  return arith::BitcastOp::create(body, loc, body.getF32Type(), bits);
}

Value i32(OpBuilder &body, Location loc, int64_t value) {
  return arith::ConstantIntOp::create(body, loc, body.getI32Type(), value);
}

void lowerMXINT8Quantize(Operation *op, int64_t axis) {
  OpBuilder builder(op);
  Location loc = op->getLoc();
  MLIRContext *context = op->getContext();
  Value input = op->getOperand(0);
  auto inputType = cast<RankedTensorType>(input.getType());
  auto result = cast<RankedTensorType>(op->getResult(0).getType());
  int64_t rank = inputType.getRank();
  int64_t k = inputType.getDimSize(axis);
  ArrayRef<int64_t> shape = inputType.getShape();
  int64_t count = ShapedType::isDynamic(k) ? k : k / 32;
  SmallVector<int64_t> blocked = withAxis<int64_t>(shape, axis, {count, 32});
  SmallVector<int64_t> blocks = withAxis<int64_t>(shape, axis, {count});
  SmallVector<OpFoldResult> sizes = tensor::getMixedSizes(builder, loc, input);
  Type f32 = builder.getF32Type();
  // K and the blocks, a dynamic K padded with zeros to whole blocks.
  OpFoldResult padded = builder.getIndexAttr(k);
  OpFoldResult blockCount = builder.getIndexAttr(count);
  Value source = input;
  if (ShapedType::isDynamic(k)) {
    Value width = arith::ConstantIndexOp::create(builder, loc, 32);
    Value n = arith::CeilDivUIOp::create(
        builder, loc,
        getValueOrCreateConstantIndexOp(builder, loc, sizes[axis]), width);
    blockCount = n;
    padded = arith::MulIOp::create(builder, loc, n, width).getResult();
    SmallVector<OpFoldResult> zeroOffsets(rank, builder.getIndexAttr(0));
    SmallVector<OpFoldResult> unitStrides(rank, builder.getIndexAttr(1));
    Value zeros = linalg::FillOp::create(
                      builder, loc,
                      ValueRange{arith::ConstantOp::create(
                          builder, loc, builder.getF32FloatAttr(0))},
                      ValueRange{tensor::EmptyOp::create(
                          builder, loc,
                          withAxis<OpFoldResult>(sizes, axis, {padded}), f32)})
                      .getResult(0);
    source = tensor::InsertSliceOp::create(builder, loc, input, zeros,
                                           zeroOffsets, sizes, unitStrides);
  }
  SmallVector<OpFoldResult> paddedSizes =
      withAxis<OpFoldResult>(sizes, axis, {padded});
  SmallVector<OpFoldResult> blockedSizes = withAxis<OpFoldResult>(
      sizes, axis, {blockCount, builder.getIndexAttr(32)});
  SmallVector<OpFoldResult> blockSizes =
      withAxis<OpFoldResult>(sizes, axis, {blockCount});
  Type i8 = builder.getI8Type();
  SmallVector<utils::IteratorType> parallel(rank + 1,
                                            utils::IteratorType::parallel);
  SmallVector<utils::IteratorType> reduce = parallel;
  reduce[axis + 1] = utils::IteratorType::reduction;

  Value values = tensor::ExpandShapeOp::create(
      builder, loc, RankedTensorType::get(blocked, f32), source,
      blockGroups(rank, axis), blockedSizes);
  Value zero =
      arith::ConstantOp::create(builder, loc, builder.getF32FloatAttr(0));
  Value maximumInit =
      linalg::FillOp::create(
          builder, loc, ValueRange{zero},
          ValueRange{tensor::EmptyOp::create(builder, loc, blockSizes, f32)})
          .getResult(0);
  Value maximum =
      linalg::GenericOp::create(
          builder, loc, maximumInit.getType(), ValueRange{values},
          ValueRange{maximumInit},
          SmallVector<AffineMap>{blockedMap(context, rank),
                                 blockMap(context, rank, axis)},
          reduce,
          [&](OpBuilder &body, Location loc, ValueRange args) {
            Value magnitude = math::AbsFOp::create(body, loc, args[0]);
            linalg::YieldOp::create(body, loc,
                                    ValueRange{arith::MaximumFOp::create(
                                        body, loc, magnitude, args[1])});
          })
          .getResult(0);

  // The byte e + 127 of a block: the biased FP32 exponent of the maximum
  // minus 6, at least 0; 127 for an all-zero block.
  SmallVector<utils::IteratorType> blockParallel(rank,
                                                 utils::IteratorType::parallel);
  Value scales =
      linalg::GenericOp::create(
          builder, loc, RankedTensorType::get(blocks, i8), ValueRange{maximum},
          ValueRange{tensor::EmptyOp::create(builder, loc, blockSizes, i8)},
          SmallVector<AffineMap>(
              2, AffineMap::getMultiDimIdentityMap(rank, context)),
          blockParallel,
          [&](OpBuilder &body, Location loc, ValueRange args) {
            Value bits =
                arith::BitcastOp::create(body, loc, body.getI32Type(), args[0]);
            Value biased =
                arith::ShRUIOp::create(body, loc, bits, i32(body, loc, 23));
            Value byte = arith::MaxSIOp::create(
                body, loc,
                arith::SubIOp::create(body, loc, biased, i32(body, loc, 6)),
                i32(body, loc, 0));
            Value empty = arith::CmpFOp::create(
                body, loc, arith::CmpFPredicate::OEQ, args[0],
                arith::ConstantOp::create(body, loc, body.getF32FloatAttr(0)));
            byte = arith::SelectOp::create(body, loc, empty,
                                           i32(body, loc, 127), byte);
            linalg::YieldOp::create(body, loc,
                                    ValueRange{arith::TruncIOp::create(
                                        body, loc, body.getI8Type(), byte)});
          })
          .getResult(0);

  Value codes =
      linalg::GenericOp::create(
          builder, loc, RankedTensorType::get(blocked, i8),
          ValueRange{values, scales},
          ValueRange{tensor::EmptyOp::create(builder, loc, blockedSizes, i8)},
          SmallVector<AffineMap>{blockedMap(context, rank),
                                 blockMap(context, rank, axis),
                                 blockedMap(context, rank)},
          parallel,
          [&](OpBuilder &body, Location loc, ValueRange args) {
            // x * 2^-e, with 2^-e = 2^((254 - byte) - 127).
            Value byte =
                arith::ExtUIOp::create(body, loc, body.getI32Type(), args[1]);
            Value multiplier = power(
                body, loc,
                arith::SubIOp::create(body, loc, i32(body, loc, 254), byte));
            Value scaled =
                arith::MulFOp::create(body, loc, args[0], multiplier);
            // Integer clamp bounds commute with rounding. Keep round and
            // conversion adjacent so RISC-V can use fcvt with RNE.
            Value clamped = arith::MinimumFOp::create(
                body, loc,
                arith::MaximumFOp::create(
                    body, loc, scaled,
                    arith::ConstantOp::create(body, loc,
                                              body.getF32FloatAttr(-127))),
                arith::ConstantOp::create(body, loc,
                                          body.getF32FloatAttr(127)));
            Value rounded = math::RoundEvenOp::create(body, loc, clamped);
            linalg::YieldOp::create(body, loc,
                                    ValueRange{arith::FPToSIOp::create(
                                        body, loc, body.getI8Type(), rounded)});
          })
          .getResult(0);
  codes = tensor::CollapseShapeOp::create(builder, loc, codes,
                                          blockGroups(rank, axis));

  SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
  OpFoldResult packedSize =
      ShapedType::isDynamic(k)
          ? OpFoldResult(arith::AddIOp::create(builder, loc,
                                               cast<Value>(padded),
                                               cast<Value>(blockCount))
                             .getResult())
          : OpFoldResult(builder.getIndexAttr(result.getDimSize(axis)));
  Value packed = tensor::EmptyOp::create(
      builder, loc, withAxis<OpFoldResult>(sizes, axis, {packedSize}), i8);
  packed = tensor::InsertSliceOp::create(builder, loc, codes, packed, offsets,
                                         paddedSizes, strides);
  offsets[axis] = padded;
  packed = tensor::InsertSliceOp::create(builder, loc, scales, packed, offsets,
                                         blockSizes, strides);
  op->getResult(0).replaceAllUsesWith(packed);
  op->erase();
}

// An integer constant of an integer type.
Value integer(OpBuilder &body, Location loc, Type type, int64_t value) {
  return arith::ConstantOp::create(body, loc, IntegerAttr::get(type, value));
}

// value / divisor (divisor > 0), the quotient of the magnitudes rounded half
// to even and given the sign of the value, as the intnorm Ball divides.
Value divideEven(OpBuilder &body, Location loc, Value value, Value divisor) {
  Type type = value.getType();
  Value zero = integer(body, loc, type, 0);
  Value one = integer(body, loc, type, 1);
  Value negative =
      arith::CmpIOp::create(body, loc, arith::CmpIPredicate::slt, value, zero);
  Value magnitude = arith::SelectOp::create(
      body, loc, negative, arith::SubIOp::create(body, loc, zero, value),
      value);
  Value quotient = arith::DivUIOp::create(body, loc, magnitude, divisor);
  Value twice = arith::ShLIOp::create(
      body, loc, arith::RemUIOp::create(body, loc, magnitude, divisor), one);
  Value odd = arith::CmpIOp::create(
      body, loc, arith::CmpIPredicate::ne,
      arith::AndIOp::create(body, loc, quotient, one), zero);
  Value up = arith::OrIOp::create(
      body, loc,
      arith::CmpIOp::create(body, loc, arith::CmpIPredicate::ugt, twice,
                            divisor),
      arith::AndIOp::create(body, loc,
                            arith::CmpIOp::create(body, loc,
                                                  arith::CmpIPredicate::eq,
                                                  twice, divisor),
                            odd));
  quotient = arith::AddIOp::create(body, loc, quotient,
                                   arith::ExtUIOp::create(body, loc, type, up));
  return arith::SelectOp::create(
      body, loc, negative, arith::SubIOp::create(body, loc, zero, quotient),
      quotient);
}

// The little-endian word of `bytes` bytes of the coefficients at `offset`.
Value coefficientWord(OpBuilder &body, Location loc, Value coefficients,
                      Value offset, int64_t bytes) {
  Type type = body.getIntegerType(bytes * 8);
  Value word = integer(body, loc, type, 0);
  for (int64_t byte = 0; byte < bytes; ++byte) {
    Value index = arith::AddIOp::create(
        body, loc, offset, arith::ConstantIndexOp::create(body, loc, byte));
    Value value = arith::ExtUIOp::create(
        body, loc, type,
        tensor::ExtractOp::create(body, loc, coefficients, ValueRange{index}));
    value = arith::ShLIOp::create(body, loc, value,
                                  integer(body, loc, type, byte * 8));
    word = arith::OrIOp::create(body, loc, word, value);
  }
  return word;
}

// The integer LayerNorm of Frontend.td's fixed_norm_mxint8, exactly as the
// intnorm Ball computes it (its BEMU model): every value in Q16 (round half
// to even), the mean and the variance divided with rounding half to even,
// the integer square root of variance + epsilon, z = (x - mean) * 2^16 /
// root, z * gamma / 2^16 + beta; then every 32 values a block with the bit
// length b of its largest magnitude: codes value / 2^(b - 7) (value *
// 2^(7 - b) below 7 bits) clamped to [-127, 127], the byte b + 104 (127 for
// an all-zero block).
LogicalResult lowerFixedNorm(FixedNormMXINT8Op op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  MLIRContext *context = op.getContext();
  Value input = op.getInput();
  Value coefficients = op.getCoefficients();
  auto inputType = cast<RankedTensorType>(input.getType());
  auto result = cast<RankedTensorType>(op.getResult().getType());
  int64_t rank = inputType.getRank();
  int64_t axis = rank - 1;
  int64_t k = inputType.getDimSize(axis);
  if (ShapedType::isDynamic(k) || k % 32)
    return op.emitOpError("normalizes rows of a static multiple of 32 values");
  ArrayRef<int64_t> shape = inputType.getShape();
  SmallVector<int64_t> blocked = withAxis<int64_t>(shape, axis, {k / 32, 32});
  SmallVector<int64_t> blocks = withAxis<int64_t>(shape, axis, {k / 32});
  SmallVector<OpFoldResult> sizes = tensor::getMixedSizes(builder, loc, input);
  SmallVector<OpFoldResult> rowSizes(sizes.begin(), sizes.end() - 1);
  SmallVector<OpFoldResult> blockedSizes = withAxis<OpFoldResult>(
      sizes, axis, {builder.getIndexAttr(k / 32), builder.getIndexAttr(32)});
  SmallVector<OpFoldResult> blockSizes =
      withAxis<OpFoldResult>(sizes, axis, {builder.getIndexAttr(k / 32)});
  Type i8 = builder.getI8Type(), i32 = builder.getI32Type();
  Type i64 = builder.getI64Type(), i128 = builder.getIntegerType(128);
  auto identity = AffineMap::getMultiDimIdentityMap(rank, context);
  auto rowIdentity = AffineMap::getMultiDimIdentityMap(rank - 1, context);
  SmallVector<AffineExpr> rows;
  for (int64_t dim = 0; dim < axis; ++dim)
    rows.push_back(getAffineDimExpr(dim, context));
  auto rowMap = AffineMap::get(rank, 0, rows, context);
  SmallVector<utils::IteratorType> parallel(rank,
                                            utils::IteratorType::parallel);
  SmallVector<utils::IteratorType> reduction = parallel;
  reduction.back() = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> rowParallel(rank - 1,
                                               utils::IteratorType::parallel);
  auto filled = [&](ArrayRef<OpFoldResult> sizes, Type type) -> Value {
    return linalg::FillOp::create(
               builder, loc, integer(builder, loc, type, 0),
               tensor::EmptyOp::create(builder, loc, sizes, type).getResult())
        .getResult(0);
  };

  // Q16, rounded half to even: x * 2^16 is exact in FP32.
  Value fixed =
      linalg::GenericOp::create(
          builder, loc, RankedTensorType::get(shape, i64), ValueRange{input},
          ValueRange{tensor::EmptyOp::create(builder, loc, sizes, i64)},
          ArrayRef<AffineMap>{identity, identity}, parallel,
          [&](OpBuilder &body, Location loc, ValueRange args) {
            Value scaled = arith::MulFOp::create(
                body, loc, args[0],
                arith::ConstantOp::create(body, loc,
                                          body.getF32FloatAttr(65536)));
            Value rounded = math::RoundEvenOp::create(body, loc, scaled);
            linalg::YieldOp::create(
                body, loc,
                ValueRange{arith::FPToSIOp::create(body, loc, i64, rounded)});
          })
          .getResult(0);
  Value sum =
      linalg::GenericOp::create(
          builder, loc, RankedTensorType::get(shape.drop_back(), i64),
          ValueRange{fixed}, ValueRange{filled(rowSizes, i64)},
          ArrayRef<AffineMap>{identity, rowMap}, reduction,
          [&](OpBuilder &body, Location loc, ValueRange args) {
            linalg::YieldOp::create(
                body, loc,
                ValueRange{arith::AddIOp::create(body, loc, args[0], args[1])});
          })
          .getResult(0);
  Value mean =
      linalg::GenericOp::create(
          builder, loc, RankedTensorType::get(shape.drop_back(), i64),
          ValueRange{sum},
          ValueRange{tensor::EmptyOp::create(builder, loc, rowSizes, i64)},
          ArrayRef<AffineMap>{rowIdentity, rowIdentity}, rowParallel,
          [&](OpBuilder &body, Location loc, ValueRange args) {
            linalg::YieldOp::create(
                body, loc,
                ValueRange{divideEven(body, loc, args[0],
                                      integer(body, loc, i64, k))});
          })
          .getResult(0);
  Value squares =
      linalg::GenericOp::create(
          builder, loc, RankedTensorType::get(shape.drop_back(), i128),
          ValueRange{fixed, mean}, ValueRange{filled(rowSizes, i128)},
          ArrayRef<AffineMap>{identity, rowMap, rowMap}, reduction,
          [&](OpBuilder &body, Location loc, ValueRange args) {
            Value centered = arith::ExtSIOp::create(
                body, loc, i128,
                arith::SubIOp::create(body, loc, args[0], args[1]));
            Value square = arith::MulIOp::create(body, loc, centered, centered);
            linalg::YieldOp::create(
                body, loc,
                ValueRange{arith::AddIOp::create(body, loc, square, args[2])});
          })
          .getResult(0);
  // The integer square root of variance + epsilon (an unsigned 64-bit
  // value): the FP64 root, then corrected by one either way.
  Value root =
      linalg::GenericOp::create(
          builder, loc, RankedTensorType::get(shape.drop_back(), i64),
          ValueRange{squares},
          ValueRange{tensor::EmptyOp::create(builder, loc, rowSizes, i64)},
          ArrayRef<AffineMap>{rowIdentity, rowIdentity}, rowParallel,
          [&](OpBuilder &body, Location loc, ValueRange args) {
            Value epsilon = coefficientWord(
                body, loc, coefficients,
                arith::ConstantIndexOp::create(body, loc, 0), 8);
            Value variance = arith::AddIOp::create(
                body, loc,
                divideEven(body, loc, args[0], integer(body, loc, i128, k)),
                arith::ExtUIOp::create(body, loc, i128, epsilon));
            Value estimate = arith::FPToUIOp::create(
                body, loc, i128,
                math::SqrtOp::create(body, loc,
                                     arith::UIToFPOp::create(body, loc,
                                                             body.getF64Type(),
                                                             variance)));
            Value one = integer(body, loc, i128, 1);
            Value above = arith::CmpIOp::create(
                body, loc, arith::CmpIPredicate::ugt,
                arith::MulIOp::create(body, loc, estimate, estimate), variance);
            estimate = arith::SelectOp::create(
                body, loc, above,
                arith::SubIOp::create(body, loc, estimate, one), estimate);
            Value next = arith::AddIOp::create(body, loc, estimate, one);
            Value below = arith::CmpIOp::create(
                body, loc, arith::CmpIPredicate::ule,
                arith::MulIOp::create(body, loc, next, next), variance);
            estimate =
                arith::SelectOp::create(body, loc, below, next, estimate);
            linalg::YieldOp::create(
                body, loc,
                ValueRange{arith::TruncIOp::create(body, loc, i64, estimate)});
          })
          .getResult(0);
  Value normalized =
      linalg::GenericOp::create(
          builder, loc, RankedTensorType::get(shape, i32),
          ValueRange{fixed, mean, root},
          ValueRange{tensor::EmptyOp::create(builder, loc, sizes, i32)},
          ArrayRef<AffineMap>{identity, rowMap, rowMap, identity}, parallel,
          [&](OpBuilder &body, Location loc, ValueRange args) {
            Value column = linalg::IndexOp::create(body, loc, axis);
            Value offset = arith::AddIOp::create(
                body, loc,
                arith::MulIOp::create(
                    body, loc, column,
                    arith::ConstantIndexOp::create(body, loc, 4)),
                arith::ConstantIndexOp::create(body, loc, 16));
            Value gamma = arith::ExtSIOp::create(
                body, loc, i64,
                coefficientWord(body, loc, coefficients, offset, 4));
            Value beta = arith::ExtSIOp::create(
                body, loc, i64,
                coefficientWord(
                    body, loc, coefficients,
                    arith::AddIOp::create(
                        body, loc, offset,
                        arith::ConstantIndexOp::create(body, loc, k * 4)),
                    4));
            Value unit = integer(body, loc, i64, 65536);
            Value z = divideEven(
                body, loc,
                arith::MulIOp::create(
                    body, loc,
                    arith::SubIOp::create(body, loc, args[0], args[1]), unit),
                args[2]);
            Value value = arith::AddIOp::create(
                body, loc,
                divideEven(body, loc,
                           arith::MulIOp::create(body, loc, z, gamma), unit),
                beta);
            linalg::YieldOp::create(
                body, loc,
                ValueRange{arith::TruncIOp::create(body, loc, i32, value)});
          })
          .getResult(0);

  // The blocks of 32 values: the bit length of the largest magnitude.
  SmallVector<utils::IteratorType> blockedParallel(
      rank + 1, utils::IteratorType::parallel);
  SmallVector<utils::IteratorType> blockedReduction = blockedParallel;
  blockedReduction[axis + 1] = utils::IteratorType::reduction;
  Value values = tensor::ExpandShapeOp::create(
      builder, loc, RankedTensorType::get(blocked, i32), normalized,
      blockGroups(rank, axis), blockedSizes);
  Value lengths =
      linalg::GenericOp::create(
          builder, loc, RankedTensorType::get(blocks, i64), ValueRange{values},
          ValueRange{filled(blockSizes, i64)},
          SmallVector<AffineMap>{blockedMap(context, rank),
                                 blockMap(context, rank, axis)},
          blockedReduction,
          [&](OpBuilder &body, Location loc, ValueRange args) {
            Value magnitude = math::AbsIOp::create(
                body, loc, arith::ExtSIOp::create(body, loc, i64, args[0]));
            linalg::YieldOp::create(body, loc,
                                    ValueRange{arith::MaxSIOp::create(
                                        body, loc, magnitude, args[1])});
          })
          .getResult(0);
  lengths =
      linalg::GenericOp::create(
          builder, loc, RankedTensorType::get(blocks, i64), ValueRange{lengths},
          ValueRange{tensor::EmptyOp::create(builder, loc, blockSizes, i64)},
          SmallVector<AffineMap>(2, identity), parallel,
          [&](OpBuilder &body, Location loc, ValueRange args) {
            linalg::YieldOp::create(
                body, loc,
                ValueRange{arith::SubIOp::create(
                    body, loc, integer(body, loc, i64, 64),
                    math::CountLeadingZerosOp::create(body, loc, args[0]))});
          })
          .getResult(0);
  Value scales =
      linalg::GenericOp::create(
          builder, loc, RankedTensorType::get(blocks, i8), ValueRange{lengths},
          ValueRange{tensor::EmptyOp::create(builder, loc, blockSizes, i8)},
          SmallVector<AffineMap>(2, identity), parallel,
          [&](OpBuilder &body, Location loc, ValueRange args) {
            Value empty =
                arith::CmpIOp::create(body, loc, arith::CmpIPredicate::eq,
                                      args[0], integer(body, loc, i64, 0));
            Value byte = arith::SelectOp::create(
                body, loc, empty, integer(body, loc, i64, 127),
                arith::AddIOp::create(body, loc, args[0],
                                      integer(body, loc, i64, 104)));
            linalg::YieldOp::create(
                body, loc,
                ValueRange{arith::TruncIOp::create(body, loc, i8, byte)});
          })
          .getResult(0);
  Value codes =
      linalg::GenericOp::create(
          builder, loc, RankedTensorType::get(blocked, i8),
          ValueRange{values, lengths},
          ValueRange{tensor::EmptyOp::create(builder, loc, blockedSizes, i8)},
          SmallVector<AffineMap>{blockedMap(context, rank),
                                 blockMap(context, rank, axis),
                                 blockedMap(context, rank)},
          blockedParallel,
          [&](OpBuilder &body, Location loc, ValueRange args) {
            Value value = arith::ExtSIOp::create(body, loc, i64, args[0]);
            Value seven = integer(body, loc, i64, 7);
            Value wide = arith::CmpIOp::create(
                body, loc, arith::CmpIPredicate::sge, args[1], seven);
            // Both shifts in range whichever is taken: [0, 63].
            Value down = arith::MaxSIOp::create(
                body, loc, arith::SubIOp::create(body, loc, args[1], seven),
                integer(body, loc, i64, 0));
            Value up = arith::MaxSIOp::create(
                body, loc, arith::SubIOp::create(body, loc, seven, args[1]),
                integer(body, loc, i64, 0));
            Value shifted =
                divideEven(body, loc, value,
                           arith::ShLIOp::create(
                               body, loc, integer(body, loc, i64, 1), down));
            Value code = arith::SelectOp::create(
                body, loc, wide, shifted,
                arith::ShLIOp::create(body, loc, value, up));
            code = arith::MinSIOp::create(
                body, loc,
                arith::MaxSIOp::create(body, loc, code,
                                       integer(body, loc, i64, -127)),
                integer(body, loc, i64, 127));
            linalg::YieldOp::create(
                body, loc,
                ValueRange{arith::TruncIOp::create(body, loc, i8, code)});
          })
          .getResult(0);
  codes = tensor::CollapseShapeOp::create(builder, loc, codes,
                                          blockGroups(rank, axis));
  SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
  Value packed = tensor::EmptyOp::create(
      builder, loc,
      withAxis<OpFoldResult>(sizes, axis,
                             {builder.getIndexAttr(result.getDimSize(axis))}),
      i8);
  packed = tensor::InsertSliceOp::create(builder, loc, codes, packed, offsets,
                                         sizes, strides);
  offsets[axis] = builder.getIndexAttr(k);
  packed = tensor::InsertSliceOp::create(builder, loc, scales, packed, offsets,
                                         blockSizes, strides);
  op.getResult().replaceAllUsesWith(packed);
  op.erase();
  return success();
}

void lowerMXINT8Dequantize(Operation *op, int64_t axis) {
  OpBuilder builder(op);
  Location loc = op->getLoc();
  MLIRContext *context = op->getContext();
  Value input = op->getOperand(0);
  auto result = cast<RankedTensorType>(op->getResult(0).getType());
  int64_t rank = result.getRank();
  int64_t k = result.getDimSize(axis);
  ArrayRef<int64_t> shape = result.getShape();
  int64_t count = ShapedType::isDynamic(k) ? k : k / 32;
  SmallVector<int64_t> blocked = withAxis<int64_t>(shape, axis, {count, 32});
  SmallVector<int64_t> blocks = withAxis<int64_t>(shape, axis, {count});
  SmallVector<OpFoldResult> packedSizes =
      tensor::getMixedSizes(builder, loc, input);
  // A dynamic K is padded: the packed axis holds 33 bytes a block.
  OpFoldResult padded = builder.getIndexAttr(k);
  OpFoldResult blockCount = builder.getIndexAttr(count);
  if (ShapedType::isDynamic(k)) {
    Value n = arith::DivUIOp::create(
        builder, loc,
        getValueOrCreateConstantIndexOp(builder, loc, packedSizes[axis]),
        arith::ConstantIndexOp::create(builder, loc, 33));
    blockCount = n;
    padded =
        arith::MulIOp::create(builder, loc, n,
                              arith::ConstantIndexOp::create(builder, loc, 32))
            .getResult();
  }
  SmallVector<OpFoldResult> sizes =
      withAxis<OpFoldResult>(packedSizes, axis, {padded});
  SmallVector<OpFoldResult> blockedSizes = withAxis<OpFoldResult>(
      packedSizes, axis, {blockCount, builder.getIndexAttr(32)});
  SmallVector<OpFoldResult> blockSizes =
      withAxis<OpFoldResult>(packedSizes, axis, {blockCount});
  Type i8 = builder.getI8Type();

  SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
  Value codes = tensor::ExtractSliceOp::create(builder, loc,
                                               RankedTensorType::get(shape, i8),
                                               input, offsets, sizes, strides);
  offsets[axis] = padded;
  Value scales = tensor::ExtractSliceOp::create(
      builder, loc, RankedTensorType::get(blocks, i8), input, offsets,
      blockSizes, strides);
  codes = tensor::ExpandShapeOp::create(
      builder, loc, RankedTensorType::get(blocked, i8), codes,
      blockGroups(rank, axis), blockedSizes);

  auto blockedType = RankedTensorType::get(blocked, result.getElementType());
  Value values =
      linalg::GenericOp::create(
          builder, loc, blockedType, ValueRange{codes, scales},
          ValueRange{tensor::EmptyOp::create(builder, loc, blockedSizes,
                                             result.getElementType())},
          SmallVector<AffineMap>{blockedMap(context, rank),
                                 blockMap(context, rank, axis),
                                 blockedMap(context, rank)},
          SmallVector<utils::IteratorType>(rank + 1,
                                           utils::IteratorType::parallel),
          [&](OpBuilder &body, Location loc, ValueRange args) {
            // code * 2^e as code * 2^e1 * 2^e2, e1 = floor(e / 2): 2^-127 is
            // no normal FP32, its halves are.
            Value exponent = arith::SubIOp::create(
                body, loc,
                arith::ExtUIOp::create(body, loc, body.getI32Type(), args[1]),
                i32(body, loc, 127));
            Value first =
                arith::ShRSIOp::create(body, loc, exponent, i32(body, loc, 1));
            Value second = arith::SubIOp::create(body, loc, exponent, first);
            Value code =
                arith::SIToFPOp::create(body, loc, body.getF32Type(), args[0]);
            Value value = arith::MulFOp::create(
                body, loc, code,
                power(body, loc,
                      arith::AddIOp::create(body, loc, first,
                                            i32(body, loc, 127))));
            value = arith::MulFOp::create(
                body, loc, value,
                power(body, loc,
                      arith::AddIOp::create(body, loc, second,
                                            i32(body, loc, 127))));
            linalg::YieldOp::create(body, loc, value);
          })
          .getResult(0);
  values = tensor::CollapseShapeOp::create(builder, loc, values,
                                           blockGroups(rank, axis));
  op->getResult(0).replaceAllUsesWith(values);
  op->erase();
}

// A standalone MXINT5 decode on the GPU (a device fuses it into its matrix
// consumer). Along axis each 32-value block has 20 code bytes (4 groups of 5
// bytes, 8 signed 5-bit codes each, little-endian), then all unsigned factors,
// then all E8M0 exponents; a value is code * factor * 2^(exponent - 131). A
// generic over (blocks, 4 groups, 8 codes) reads the 5 bytes of its group.
LogicalResult lowerMXINT5Dequantize(Operation *op, int64_t axis) {
  OpBuilder builder(op);
  Location loc = op->getLoc();
  MLIRContext *context = op->getContext();
  Value input = op->getOperand(0);
  auto result = cast<RankedTensorType>(op->getResult(0).getType());
  int64_t rank = result.getRank();
  int64_t k = result.getDimSize(axis);
  if (ShapedType::isDynamic(k) || k % 32)
    return op->emitError("GPU INT5 decoding needs a static K of whole blocks");
  int64_t count = k / 32;
  ArrayRef<int64_t> shape = result.getShape();
  Type i8 = builder.getI8Type();
  SmallVector<OpFoldResult> sizes = tensor::getMixedSizes(builder, loc, input);
  SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
  // The code bytes as (blocks, 4, 5) at axis.
  Value codes = tensor::ExtractSliceOp::create(
      builder, loc,
      RankedTensorType::get(withAxis<int64_t>(shape, axis, {count * 20}), i8),
      input, offsets,
      withAxis<OpFoldResult>(sizes, axis, {builder.getIndexAttr(count * 20)}),
      strides);
  SmallVector<ReassociationIndices> groups;
  for (int64_t dim = 0, expanded = 0; dim < rank; ++dim)
    if (dim == axis) {
      groups.push_back({expanded, expanded + 1, expanded + 2});
      expanded += 3;
    } else {
      groups.push_back({expanded++});
    }
  SmallVector<int64_t> grouped = withAxis<int64_t>(shape, axis, {count, 4, 5});
  SmallVector<OpFoldResult> groupedSizes = withAxis<OpFoldResult>(
      sizes, axis,
      {builder.getIndexAttr(count), builder.getIndexAttr(4),
       builder.getIndexAttr(5)});
  codes = tensor::ExpandShapeOp::create(builder, loc,
                                        RankedTensorType::get(grouped, i8),
                                        codes, groups, groupedSizes);
  // Byte j of every group: (blocks, 4) at axis.
  SmallVector<int64_t> perGroup = withAxis<int64_t>(shape, axis, {count, 4});
  SmallVector<Value> inputs;
  for (int64_t byte = 0; byte < 5; ++byte) {
    SmallVector<OpFoldResult> at(rank + 2, builder.getIndexAttr(0));
    at[axis + 2] = builder.getIndexAttr(byte);
    SmallVector<OpFoldResult> extent = groupedSizes;
    extent[axis + 2] = builder.getIndexAttr(1);
    inputs.push_back(tensor::ExtractSliceOp::create(
        builder, loc, RankedTensorType::get(perGroup, i8), codes, at, extent,
        SmallVector<OpFoldResult>(rank + 2, builder.getIndexAttr(1))));
  }
  // The factors and the exponents: (blocks) at axis.
  for (int64_t part = 0; part < 2; ++part) {
    offsets[axis] = builder.getIndexAttr(count * (20 + part));
    inputs.push_back(tensor::ExtractSliceOp::create(
        builder, loc,
        RankedTensorType::get(withAxis<int64_t>(shape, axis, {count}), i8),
        input, offsets,
        withAxis<OpFoldResult>(sizes, axis, {builder.getIndexAttr(count)}),
        strides));
  }
  auto dropping = [&](ArrayRef<int64_t> dropped) {
    SmallVector<AffineExpr> results;
    for (int64_t dim = 0; dim < rank + 2; ++dim)
      if (!llvm::is_contained(dropped, dim))
        results.push_back(getAffineDimExpr(dim, context));
    return AffineMap::get(rank + 2, 0, results, context);
  };
  SmallVector<AffineMap> maps(5, dropping({axis + 2}));
  maps.append(2, dropping({axis + 1, axis + 2}));
  maps.push_back(AffineMap::getMultiDimIdentityMap(rank + 2, context));
  SmallVector<int64_t> decoded = withAxis<int64_t>(shape, axis, {count, 4, 8});
  SmallVector<OpFoldResult> decodedSizes = withAxis<OpFoldResult>(
      sizes, axis,
      {builder.getIndexAttr(count), builder.getIndexAttr(4),
       builder.getIndexAttr(8)});
  Value values =
      linalg::GenericOp::create(
          builder, loc, RankedTensorType::get(decoded, result.getElementType()),
          inputs,
          ValueRange{tensor::EmptyOp::create(builder, loc, decodedSizes,
                                             result.getElementType())},
          maps,
          SmallVector<utils::IteratorType>(rank + 2,
                                           utils::IteratorType::parallel),
          [&](OpBuilder &body, Location loc, ValueRange args) {
            Type i64 = body.getI64Type();
            Value word = arith::ConstantIntOp::create(body, loc, i64, 0);
            for (int64_t byte = 0; byte < 5; ++byte)
              word = arith::OrIOp::create(
                  body, loc, word,
                  arith::ShLIOp::create(
                      body, loc,
                      arith::ExtUIOp::create(body, loc, i64, args[byte]),
                      arith::ConstantIntOp::create(body, loc, i64, 8 * byte)));
            Value position = arith::IndexCastOp::create(
                body, loc, i64, linalg::IndexOp::create(body, loc, axis + 2));
            Value code = arith::AndIOp::create(
                body, loc,
                arith::ShRUIOp::create(
                    body, loc, word,
                    arith::MulIOp::create(
                        body, loc, position,
                        arith::ConstantIntOp::create(body, loc, i64, 5))),
                arith::ConstantIntOp::create(body, loc, i64, 31));
            Value sign = arith::ConstantIntOp::create(body, loc, i64, 16);
            code = arith::SubIOp::create(
                body, loc, arith::XOrIOp::create(body, loc, code, sign), sign);
            Value product = arith::MulIOp::create(
                body, loc, code,
                arith::ExtUIOp::create(body, loc, i64, args[5]));
            // 2^(e - 131) as 2^e1 * 2^e2, e1 = floor((e - 131) / 2): each half
            // is a normal FP32.
            Value exponent = arith::SubIOp::create(
                body, loc,
                arith::ExtUIOp::create(body, loc, body.getI32Type(), args[6]),
                i32(body, loc, 131));
            Value first =
                arith::ShRSIOp::create(body, loc, exponent, i32(body, loc, 1));
            Value second = arith::SubIOp::create(body, loc, exponent, first);
            Value value =
                arith::SIToFPOp::create(body, loc, body.getF32Type(), product);
            value = arith::MulFOp::create(
                body, loc, value,
                power(body, loc,
                      arith::AddIOp::create(body, loc, first,
                                            i32(body, loc, 127))));
            value = arith::MulFOp::create(
                body, loc, value,
                power(body, loc,
                      arith::AddIOp::create(body, loc, second,
                                            i32(body, loc, 127))));
            linalg::YieldOp::create(body, loc, value);
          })
          .getResult(0);
  values =
      tensor::CollapseShapeOp::create(builder, loc, result, values, groups);
  op->getResult(0).replaceAllUsesWith(values);
  op->erase();
  return success();
}

// The private declaration of a trace runtime function, at the top of the
// module.
void declare(ModuleOp module, StringRef name, TypeRange arguments) {
  if (module.lookupSymbol<func::FuncOp>(name))
    return;
  OpBuilder builder(module.getBodyRegion());
  auto function = func::FuncOp::create(builder, module.getLoc(), name,
                                       builder.getFunctionType(arguments, {}));
  function.setPrivate();
  function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
}

// The id, the depth of the id path and its four components.
SmallVector<Value> pathArguments(OpBuilder &builder, Location loc, int64_t id,
                                 ArrayAttr idPath) {
  SmallVector<Value> values;
  values.push_back(arith::ConstantIntOp::create(builder, loc, id, 64));
  values.push_back(
      arith::ConstantIntOp::create(builder, loc, idPath.size(), 64));
  for (size_t i = 0; i < 4; ++i)
    values.push_back(arith::ConstantIntOp::create(
        builder, loc,
        i < idPath.size() ? cast<IntegerAttr>(idPath[i]).getInt() : -1, 64));
  return values;
}

void declareCycle(ModuleOp module) {
  Type i64 = IntegerType::get(module.getContext(), 64);
  SmallVector<Type> path(6, i64);
  declare(module, "buddyTraceCycleStartPath", path);
  declare(module, "buddyTraceCycleEndPath", path);
}

LogicalResult lowerTensorTrace(TraceEndOp op) {
  auto type = dyn_cast<RankedTensorType>(op.getInput().getType());
  if (!type || !type.hasStaticShape())
    return op.emitOpError("traces a tensor of no static shape");
  Type element = type.getElementType();
  StringRef name;
  if (element.isF32())
    name = "buddyTraceTensorF32Path";
  else if (element.isBF16())
    name = "buddyTraceTensorBF16Path";
  else if (element.isInteger(8))
    name = "buddyTraceTensorI8Path";
  else if (element.isInteger(32))
    name = "buddyTraceTensorI32Path";
  else
    return op.emitOpError("traces a tensor that is not f32, bf16, i8 or i32");
  auto module = op->getParentOfType<ModuleOp>();
  Type i64 = IntegerType::get(module.getContext(), 64);
  auto flatType = MemRefType::get({ShapedType::kDynamic}, element);
  SmallVector<Type> arguments(6, i64);
  arguments.push_back(flatType);
  declare(module, name, arguments);

  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value buffer = bufferization::ToBufferOp::create(
      builder, loc, MemRefType::get(type.getShape(), element), op.getInput(),
      /*read_only=*/true);
  if (type.getRank() != 1) {
    ReassociationIndices all;
    for (int64_t dim = 0; dim < type.getRank(); ++dim)
      all.push_back(dim);
    buffer = memref::CollapseShapeOp::create(
        builder, loc, MemRefType::get({type.getNumElements()}, element), buffer,
        SmallVector<ReassociationIndices>{all});
  }
  Value flat = memref::CastOp::create(builder, loc, flatType, buffer);
  SmallVector<Value> values =
      pathArguments(builder, loc, op.getId(), op.getIdPath());
  values.push_back(flat);
  func::CallOp::create(builder, loc, name, TypeRange{}, values);
  return success();
}

class LowerFrontendFakeOpPass
    : public PassWrapper<LowerFrontendFakeOpPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerFrontendFakeOpPass)

  LowerFrontendFakeOpPass() = default;
  LowerFrontendFakeOpPass(const LowerFrontendFakeOpPass &other)
      : PassWrapper(other) {}

  // On a GPU target the values live in device memory, which the trace
  // runtime on the host cannot read: a tensor trace is an error there.
  Option<bool> gpu{*this, "gpu", llvm::cl::desc("Values live in GPU memory")};

  StringRef getArgument() const final { return "lower-frontend-fake-op"; }
  StringRef getDescription() const final {
    return "Compute the frontend operations the design left in software";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, linalg::LinalgDialect, math::MathDialect,
                    memref::MemRefDialect, tensor::TensorDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Shape queries must not keep an otherwise unused quantization alive.
    RewritePatternSet shapes(&getContext());
    memref::populateResolveRankedShapedTypeResultDimsPatterns(shapes);
    if (failed(applyPatternsGreedily(module, std::move(shapes))))
      return signalPassFailure();
    SmallVector<Operation *> quantizations;
    SmallVector<TraceStartOp> starts;
    SmallVector<TraceEndOp> ends;
    module.walk([&](Operation *op) {
      if (auto start = dyn_cast<TraceStartOp>(op))
        starts.push_back(start);
      else if (auto end = dyn_cast<TraceEndOp>(op))
        ends.push_back(end);
      else if (isa<FrontendDialect>(op->getDialect()))
        quantizations.push_back(op);
    });

    for (Operation *op : quantizations) {
      if (auto decode = dyn_cast<MXINT5ToFP32Op>(op)) {
        if (!gpu) {
          op->emitError(
              "INT5 weight decoding must be fused with its matrix consumer");
          return signalPassFailure();
        }
        if (failed(lowerMXINT5Dequantize(op, decode.getAxis())))
          return signalPassFailure();
      } else if (isa<MXFP4ToFP32Op>(op)) {
        op->emitError(
            "MXFP4 weight decoding must be fused with its matrix consumer");
        return signalPassFailure();
      } else if (isa<IntegerSoftmaxOp>(op)) {
        if (!gpu) {
          op->emitError(
              "integer softmax requires an integer row operator lowering");
          return signalPassFailure();
        }
        if (failed(mlir::buddy::lowerNonlinear(op)))
          return signalPassFailure();
      } else if (auto norm = dyn_cast<FixedNormMXINT8Op>(op)) {
        if (!gpu) {
          op->emitError("integer normalization requires the design's lowering");
          return signalPassFailure();
        }
        if (failed(lowerFixedNorm(norm)))
          return signalPassFailure();
      } else if (isa<MXINT8ActivationOp>(op)) {
        if (failed(mlir::buddy::lowerMxint8Activation(op)))
          return signalPassFailure();
      } else if (isa<FP32ToInt8Op, Int32ToInt8Op, FP32ToInt32Op>(op))
        lower(op, true);
      else if (isa<Int8ToFP32Op, Int32ToFP32Op>(op))
        lower(op, false);
      else if (auto add = dyn_cast<ScaledAddI8Op>(op)) {
        if (failed(lowerScaledAdd(add)))
          return signalPassFailure();
      } else if (auto lookup = dyn_cast<LookupI8Op>(op))
        lowerLookup(lookup);
      else if (auto concat = dyn_cast<ConcatColumnsOp>(op)) {
        if (failed(mlir::buddy::lowerConcatColumns(concat)))
          return signalPassFailure();
      } else if (auto gather = dyn_cast<GatherRowsOp>(op)) {
        if (failed(mlir::buddy::lowerGatherRows(gather)))
          return signalPassFailure();
      } else if (isa<LookupI32Op, SoftmaxI8Op, RowMaxU32Op>(op)) {
        if (failed(mlir::buddy::lowerNonlinear(op)))
          return signalPassFailure();
      } else if (isa<FixedRequantOp, FixedUnaryOp, FixedBinaryOp, FixedMeanOp>(
                     op)) {
        if (failed(mlir::buddy::lowerFixedPoint(op)))
          return signalPassFailure();
      } else if (auto quantize = dyn_cast<FP32ToMXINT8Op>(op))
        lowerMXINT8Quantize(op, quantize.getAxis());
      else
        lowerMXINT8Dequantize(op, cast<MXINT8ToFP32Op>(op).getAxis());
    }

    for (TraceStartOp start : starts) {
      if (start.getIdPath().size() > 4) {
        start.emitOpError("has an id path deeper than four");
        return signalPassFailure();
      }
      if (start.getCycle()) {
        declareCycle(module);
        OpBuilder builder(start);
        func::CallOp::create(builder, start.getLoc(),
                             "buddyTraceCycleStartPath", TypeRange{},
                             pathArguments(builder, start.getLoc(),
                                           start.getId(), start.getIdPath()));
      }
      start.erase();
    }

    for (TraceEndOp end : ends) {
      if (end.getIdPath().size() > 4) {
        end.emitOpError("has an id path deeper than four");
        return signalPassFailure();
      }
      if (end.getCycle()) {
        declareCycle(module);
        OpBuilder builder(end);
        func::CallOp::create(
            builder, end.getLoc(), "buddyTraceCycleEndPath", TypeRange{},
            pathArguments(builder, end.getLoc(), end.getId(), end.getIdPath()));
      }
      if (end.getTensor() && gpu) {
        end.emitOpError("traces a tensor in GPU memory, which the host "
                        "trace runtime cannot read");
        return signalPassFailure();
      }
      if (end.getTensor() && failed(lowerTensorTrace(end)))
        return signalPassFailure();
      end.getOutput().replaceAllUsesWith(end.getInput());
      end.erase();
    }
  }
};
} // namespace

namespace mlir::buddy {
void registerLowerFrontendFakeOpPass() {
  PassRegistration<LowerFrontendFakeOpPass>();
}
} // namespace mlir::buddy

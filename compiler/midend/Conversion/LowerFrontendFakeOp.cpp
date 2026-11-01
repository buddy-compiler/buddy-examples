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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"

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
            // conversion adjacent so RISC-V can use fcvt.w.s with RNE.
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
                    memref::MemRefDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
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
      if (isa<FP32ToInt8Op>(op))
        lower(op, true);
      else if (isa<Int8ToFP32Op, Int32ToFP32Op>(op))
        lower(op, false);
      else if (auto quantize = dyn_cast<FP32ToMXINT8Op>(op))
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

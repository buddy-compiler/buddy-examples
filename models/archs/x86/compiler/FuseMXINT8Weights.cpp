//===- FuseMXINT8Weights.cpp - MXINT8 weights read packed by their users --===//
//
// The x86 backend's own pass (backends share no code).
//
// A weight the frontend packed to MXINT8 offline (a function argument, N x K
// packed along K) is read by mxint8_to_fp32 and then by one contraction (a
// Linear's matmul). lower-frontend-fake-op would restore the whole weight to an
// FP32 tensor first. This pass restores every weight value exactly once and
// never holds the whole FP32 weight:
//
//   - Each output row reads the weight once (M = 1, a decode step): the
//     contraction becomes a linalg.generic that reads the codes and the
//     exponent bytes and restores each value where it loads it.
//   - Several rows read it (M > 1): the contraction runs over tiles of `tile`
//     weight rows (an scf.for); each tile is restored once into an FP32 tile
//     of at most `budget` bytes, which every row of the tile then reads. The
//     budget (256 KiB) keeps the tile in the L2 cache of one core next to the
//     rows it multiplies, so the M reads of a tile hit L2; the tile row count
//     is the largest power of two dividing N within it.
//
// Both forms keep each output's accumulation over K in its order and restore
// the values of mxint8_to_fp32: code * 2^e, with e the exponent byte of the
// code's block of 32 along K minus 127, formed as code * 2^e1 * 2^e2
// (e1 = floor(e / 2)) from exponent bits: bit-identical to the unfused graph.
//
// Run before lower-frontend-fake-op, which computes every other frontend
// operation.
//
//===----------------------------------------------------------------------===//

#include "Frontend/FrontendOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace ::buddy::frontend;

namespace {
// The FP32 bytes of one restored weight tile.
constexpr int64_t budget = 256 << 10;

Value i32(OpBuilder &body, Location loc, int64_t value) {
  return arith::ConstantIntOp::create(body, loc, body.getI32Type(), value);
}

// 2^(biased - 127) for an i32 biased exponent in [1, 254].
Value power(OpBuilder &body, Location loc, Value biased) {
  Value bits = arith::ShLIOp::create(body, loc, biased, i32(body, loc, 23));
  return arith::BitcastOp::create(body, loc, body.getF32Type(), bits);
}

// The value of an MXINT8 code with its exponent byte.
Value restore(OpBuilder &body, Location loc, Value code, Value byte) {
  Value exponent = arith::SubIOp::create(
      body, loc, arith::ExtUIOp::create(body, loc, body.getI32Type(), byte),
      i32(body, loc, 127));
  Value first = arith::ShRSIOp::create(body, loc, exponent, i32(body, loc, 1));
  Value second = arith::SubIOp::create(body, loc, exponent, first);
  Value value = arith::SIToFPOp::create(body, loc, body.getF32Type(), code);
  value = arith::MulFOp::create(
      body, loc, value,
      power(body, loc,
            arith::AddIOp::create(body, loc, first, i32(body, loc, 127))));
  return arith::MulFOp::create(
      body, loc, value,
      power(body, loc,
            arith::AddIOp::create(body, loc, second, i32(body, loc, 127))));
}

// A copy of `generic` on `inputs` and `outputs` (same maps and body); the
// value of input `index` is computed from the generic's block arguments by
// `input`, given the arguments of the new body.
linalg::GenericOp
copy(RewriterBase &rewriter, Location loc, linalg::GenericOp generic,
     ValueRange inputs, ValueRange outputs, ArrayRef<AffineMap> maps,
     unsigned index,
     function_ref<Value(OpBuilder &, Location, ValueRange)> input) {
  Block &old = generic.getRegion().front();
  unsigned count = generic.getNumDpsInputs();
  SmallVector<Type> types;
  for (Value output : outputs)
    types.push_back(output.getType());
  return linalg::GenericOp::create(
      rewriter, loc, types, inputs, outputs, maps,
      generic.getIteratorTypesArray(),
      [&](OpBuilder &body, Location loc, ValueRange args) {
        IRMapping mapping;
        unsigned extra = inputs.size() - count;
        for (unsigned i = 0; i < count; ++i)
          mapping.map(old.getArgument(i),
                      i == index ? input(body, loc, args) : args[i]);
        for (unsigned i = count; i < old.getNumArguments(); ++i)
          mapping.map(old.getArgument(i), args[i + extra]);
        for (Operation &inner : old.without_terminator())
          body.clone(inner, mapping);
        SmallVector<Value> yielded;
        for (Value v : old.getTerminator()->getOperands())
          yielded.push_back(mapping.lookupOrDefault(v));
        linalg::YieldOp::create(body, loc, yielded);
      });
}

LogicalResult fuse(MXINT8ToFP32Op op, linalg::LinalgOp consumer) {
  IRRewriter rewriter(op->getContext());
  rewriter.setInsertionPoint(consumer);
  Location loc = op->getLoc();
  MLIRContext *context = op->getContext();
  Value packed = op.getInput();
  auto result = cast<RankedTensorType>(op.getResult().getType());
  if (result.getRank() != 2 || op.getAxis() != 1 || !result.hasStaticShape())
    return op->emitOpError("packs no static N x K weight along K");
  int64_t n = result.getDimSize(0);
  int64_t k = result.getDimSize(1);

  linalg::GenericOp generic;
  if (auto existing = dyn_cast<linalg::GenericOp>(consumer.getOperation())) {
    generic = existing;
  } else {
    FailureOr<linalg::GenericOp> generalized =
        linalg::generalizeNamedOp(rewriter, consumer);
    if (failed(generalized))
      return consumer->emitOpError("cannot be generalized");
    generic = *generalized;
  }
  if (generic.getNumDpsInits() != 1)
    return generic->emitOpError("has more than one output");
  rewriter.setInsertionPoint(generic);

  unsigned inputs = generic.getNumDpsInputs();
  unsigned index = inputs;
  for (unsigned i = 0; i < inputs; ++i)
    if (generic.getDpsInputOperand(i)->get() == op.getResult())
      index = i;
  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  AffineMap map = maps[index];
  auto rowDim = dyn_cast<AffineDimExpr>(map.getResult(0));
  if (!rowDim)
    return generic->emitOpError("reads the weight rows through an expression");
  unsigned row = rowDim.getPosition();

  // M: the output elements one weight value reaches.
  SmallVector<int64_t> ranges = generic.getStaticLoopRanges();
  int64_t rows = 1;
  for (auto [dim, type] : llvm::enumerate(generic.getIteratorTypesArray()))
    if (type == utils::IteratorType::parallel && !map.isFunctionOfDim(dim))
      rows = ShapedType::isDynamic(ranges[dim]) || ShapedType::isDynamic(rows)
                 ? ShapedType::kDynamic
                 : rows * ranges[dim];

  SmallVector<OpFoldResult> zero(2, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> one(2, rewriter.getIndexAttr(1));
  Type i8 = rewriter.getI8Type();
  if (rows == 1) {
    // One read per weight value: restore it where the contraction loads it.
    Value codes = tensor::ExtractSliceOp::create(
        rewriter, loc, RankedTensorType::get({n, k}, i8), packed, zero,
        getAsIndexOpFoldResult(context, {n, k}), one);
    Value scales = tensor::ExtractSliceOp::create(
        rewriter, loc, RankedTensorType::get({n, k / 32}, i8), packed,
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(0),
                                  rewriter.getIndexAttr(k)},
        getAsIndexOpFoldResult(context, {n, k / 32}), one);
    SmallVector<Value> newInputs(generic.getDpsInputs());
    newInputs[index] = codes;
    newInputs.push_back(scales);
    SmallVector<AffineMap> newMaps(maps.begin(), maps.begin() + inputs);
    newMaps.push_back(AffineMap::get(
        map.getNumDims(), 0, {map.getResult(0), map.getResult(1).floorDiv(32)},
        context));
    newMaps.push_back(maps[inputs]);
    linalg::GenericOp fused =
        copy(rewriter, loc, generic, newInputs, generic.getDpsInits(), newMaps,
             index, [&](OpBuilder &body, Location loc, ValueRange args) {
               return restore(body, loc, args[index], args[inputs]);
             });
    rewriter.replaceOp(generic, fused.getResults());
    rewriter.eraseOp(op);
    return success();
  }

  // Several reads per weight value: restore tiles of `tile` rows once.
  int64_t tile = 1;
  while (n % (tile * 2) == 0 && tile * 2 * k * 4 <= budget)
    tile *= 2;
  Value output = generic.getDpsInits()[0];
  auto loop = scf::ForOp::create(
      rewriter, loc, arith::ConstantIndexOp::create(rewriter, loc, 0),
      arith::ConstantIndexOp::create(rewriter, loc, n),
      arith::ConstantIndexOp::create(rewriter, loc, tile), ValueRange{output});
  rewriter.setInsertionPointToStart(loop.getBody());
  Value j = loop.getInductionVar();
  Value accumulator = loop.getRegionIterArgs()[0];

  // The FP32 tile of weight rows [j, j + tile).
  Value codes = tensor::ExtractSliceOp::create(
      rewriter, loc, RankedTensorType::get({tile, k}, i8), packed,
      SmallVector<OpFoldResult>{j, rewriter.getIndexAttr(0)},
      getAsIndexOpFoldResult(context, {tile, k}), one);
  Value scales = tensor::ExtractSliceOp::create(
      rewriter, loc, RankedTensorType::get({tile, k / 32}, i8), packed,
      SmallVector<OpFoldResult>{j, rewriter.getIndexAttr(k)},
      getAsIndexOpFoldResult(context, {tile, k / 32}), one);
  AffineMap identity = AffineMap::getMultiDimIdentityMap(2, context);
  AffineMap block = AffineMap::get(
      2, 0,
      {getAffineDimExpr(0, context), getAffineDimExpr(1, context).floorDiv(32)},
      context);
  Value weights =
      linalg::GenericOp::create(
          rewriter, loc,
          RankedTensorType::get({tile, k}, rewriter.getF32Type()),
          ValueRange{codes, scales},
          ValueRange{tensor::EmptyOp::create(rewriter, loc,
                                             ArrayRef<int64_t>{tile, k},
                                             rewriter.getF32Type())},
          SmallVector<AffineMap>{identity, block, identity},
          SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
          [&](OpBuilder &body, Location loc, ValueRange args) {
            linalg::YieldOp::create(body, loc,
                                    restore(body, loc, args[0], args[1]));
          })
          .getResult(0);

  // Every operand restricted to the tile's rows: the dimensions indexed by
  // the weight row loop.
  auto restrict = [&](Value value, AffineMap operandMap) {
    auto type = cast<RankedTensorType>(value.getType());
    SmallVector<OpFoldResult> offsets, sizes, strides;
    SmallVector<int64_t> shape;
    for (auto [position, expr] : llvm::enumerate(operandMap.getResults())) {
      auto dim = dyn_cast<AffineDimExpr>(expr);
      strides.push_back(rewriter.getIndexAttr(1));
      if (dim && dim.getPosition() == row) {
        offsets.push_back(j);
        sizes.push_back(rewriter.getIndexAttr(tile));
        shape.push_back(tile);
      } else {
        offsets.push_back(rewriter.getIndexAttr(0));
        sizes.push_back(tensor::getMixedSize(rewriter, loc, value, position));
        shape.push_back(type.getDimSize(position));
      }
    }
    return std::make_tuple(
        tensor::ExtractSliceOp::create(
            rewriter, loc, RankedTensorType::get(shape, type.getElementType()),
            value, offsets, sizes, strides)
            .getResult(),
        offsets, sizes, strides);
  };
  SmallVector<Value> tileInputs;
  for (unsigned i = 0; i < inputs; ++i)
    tileInputs.push_back(
        i == index ? weights
                   : std::get<0>(restrict(generic.getDpsInputs()[i], maps[i])));
  auto [slice, offsets, sizes, strides] = restrict(accumulator, maps[inputs]);
  linalg::GenericOp tiled =
      copy(rewriter, loc, generic, tileInputs, ValueRange{slice}, maps, inputs,
           [](OpBuilder &, Location, ValueRange) { return Value(); });
  Value updated = tensor::InsertSliceOp::create(
      rewriter, loc, tiled.getResult(0), accumulator, offsets, sizes, strides);
  scf::YieldOp::create(rewriter, loc, updated);

  rewriter.replaceOp(generic, loop.getResults());
  rewriter.eraseOp(op);
  return success();
}

class FuseMXINT8WeightsPass
    : public PassWrapper<FuseMXINT8WeightsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseMXINT8WeightsPass)

  StringRef getArgument() const final { return "fuse-mxint8-weights"; }
  StringRef getDescription() const final {
    return "Restore each MXINT8 weight value once, inside the contraction";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    // A packed weight (a function argument) read by one contraction.
    SmallVector<std::pair<MXINT8ToFP32Op, linalg::LinalgOp>> weights;
    getOperation().walk([&](MXINT8ToFP32Op op) {
      if (!isa<BlockArgument>(op.getInput()) || !op->hasOneUse())
        return;
      auto consumer = dyn_cast<linalg::LinalgOp>(*op->user_begin());
      if (consumer && consumer.getNumReductionLoops() > 0 &&
          consumer.isDpsInput(&*op->use_begin()))
        weights.push_back({op, consumer});
    });
    for (auto [op, consumer] : weights)
      if (failed(fuse(op, consumer)))
        return signalPassFailure();
  }
};
} // namespace

namespace mlir::x86 {
void registerFuseMXINT8WeightsPass() {
  PassRegistration<FuseMXINT8WeightsPass>();
}
} // namespace mlir::x86

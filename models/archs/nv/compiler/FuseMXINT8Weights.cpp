//===- FuseMXINT8Weights.cpp - MXINT8 weights read packed by their users --===//
//
// The NVIDIA backend's own pass (backends share no code).
//
// A weight the frontend packed to MXINT8 offline (a function argument, N x K
// packed along K) is read by mxint8_to_fp32 and then by one contraction (a
// Linear's matmul). lower-frontend-fake-op would restore the whole weight to an
// FP32 tensor first; this pass instead rewrites the contraction into a
// linalg.generic that reads the codes and the exponent bytes of the packed
// weight and restores each value where the contraction loads it, so no FP32
// copy of the weight exists. linalg-to-gpu-kernels stages these restored
// values in shared memory for reuse by the output rows of each block.
// The values are those
// of mxint8_to_fp32: code * 2^e, with e the exponent byte of the code's block
// of 32 along K minus 127, formed as code * 2^e1 * 2^e2 (e1 = floor(e / 2))
// from exponent bits: bit-identical to the unfused graph.
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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace ::buddy::frontend;

namespace {
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

  SmallVector<OpFoldResult> zero(2, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> one(2, rewriter.getIndexAttr(1));
  Type i8 = rewriter.getI8Type();
  // Restore each value where the contraction loads it.
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

class FuseMXINT8WeightsPass
    : public PassWrapper<FuseMXINT8WeightsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseMXINT8WeightsPass)

  StringRef getArgument() const final { return "fuse-mxint8-weights"; }
  StringRef getDescription() const final {
    return "Read the MXINT8 weights packed inside the contractions using them";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
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

namespace mlir::nv {
void registerFuseMXINT8WeightsPass() {
  PassRegistration<FuseMXINT8WeightsPass>();
}
} // namespace mlir::nv

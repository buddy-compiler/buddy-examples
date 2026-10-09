#include "RVVTensorKernels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include <limits>
using namespace mlir;
namespace mlir::buddy::rvv {
// Strip scalar region arguments and reshapes that only add unit dimensions.
Value source(Value value) {
  while (true) {
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      auto generic =
          dyn_cast_or_null<linalg::GenericOp>(arg.getOwner()->getParentOp());
      if (!generic || arg.getArgNumber() >= generic.getNumDpsInputs())
        return value;
      Value input = generic.getDpsInputOperand(arg.getArgNumber())->get();
      auto inputType = cast<RankedTensorType>(input.getType());
      auto map = generic.getIndexingMapsArray()[arg.getArgNumber()];
      for (auto [index, expr] : llvm::enumerate(map.getResults())) {
        if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
          if (dim.getPosition() != index)
            return value;
        } else if (auto constant = dyn_cast<AffineConstantExpr>(expr)) {
          if (constant.getValue() != 0 || inputType.getShape()[index] != 1)
            return value;
        } else {
          return value;
        }
      }
      value = input;
      continue;
    }
    if (auto transpose = value.getDefiningOp<linalg::TransposeOp>()) {
      auto shape =
          cast<RankedTensorType>(transpose.getInput().getType()).getShape();
      int64_t previous = -1;
      for (int64_t dimension : transpose.getPermutation()) {
        if (shape[dimension] == 1)
          continue;
        if (dimension < previous)
          return value;
        previous = dimension;
      }
      value = transpose.getInput();
      continue;
    }
    if (auto expand = value.getDefiningOp<tensor::ExpandShapeOp>()) {
      auto shape =
          cast<RankedTensorType>(expand.getResult().getType()).getShape();
      for (auto group : expand.getReassociationIndices()) {
        unsigned nonUnit = 0;
        for (int64_t dimension : group)
          nonUnit += shape[dimension] != 1;
        if (nonUnit > 1)
          return value;
      }
      value = expand.getSrc();
      continue;
    }
    return value;
  }
}

Value expression(Value value) {
  value = source(value);
  if (auto generic = value.getDefiningOp<linalg::GenericOp>()) {
    if (generic.getNumResults() != 1 ||
        generic.getNumParallelLoops() != generic.getNumLoops() ||
        !generic.getIndexingMapsArray().back().isIdentity() ||
        generic.getLibraryCallAttr())
      return value;
    return expression(cast<linalg::YieldOp>(generic.getBody()->getTerminator())
                          .getValues()[0]);
  }
  return value;
}

Operation *operation(Value value, StringRef name) {
  Operation *op = expression(value).getDefiningOp();
  return op && op->getName().getStringRef() == name ? op : nullptr;
}

bool constant(Value value, double expected) {
  auto op = expression(value).getDefiningOp<arith::ConstantOp>();
  if (!op)
    return false;
  if (auto number = dyn_cast<FloatAttr>(op.getValue()))
    return number.getValueAsDouble() == expected;
  if (auto number = dyn_cast<IntegerAttr>(op.getValue()))
    return number.getInt() == expected;
  if (auto number = dyn_cast<DenseFPElementsAttr>(op.getValue()))
    return number.isSplat() &&
           number.getSplatValue<APFloat>().convertToDouble() == expected;
  return false;
}

namespace {
std::optional<uint32_t> floatBits(Value value) {
  auto op = expression(value).getDefiningOp<arith::ConstantOp>();
  if (!op)
    return std::nullopt;
  if (auto attr = dyn_cast<FloatAttr>(op.getValue());
      attr && attr.getType().isF32())
    return attr.getValue().bitcastToAPInt().getZExtValue();
  if (auto attr = dyn_cast<DenseFPElementsAttr>(op.getValue());
      attr && attr.isSplat() && attr.getElementType().isF32())
    return attr.getSplatValue<APFloat>().bitcastToAPInt().getZExtValue();
  return std::nullopt;
}

// Only last-axis reductions with the original ordered scalar operation match.
linalg::ReduceOp reduction(Value value, StringRef name, double seed) {
  auto reduce = source(value).getDefiningOp<linalg::ReduceOp>();
  if (!reduce || reduce.getNumResults() != 1 ||
      reduce.getInputs().size() != 1 || reduce.getDimensions().size() != 1)
    return {};
  auto input = dyn_cast<RankedTensorType>(reduce.getInputs()[0].getType());
  if (!input || reduce.getDimensions()[0] != input.getRank() - 1)
    return {};
  auto fill = reduce.getInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fill || !constant(fill.getInputs()[0], seed) ||
      (seed == 0.0 && floatBits(fill.getInputs()[0]) != 0u))
    return {};
  Block &body = reduce.getRegion().front();
  auto result = cast<linalg::YieldOp>(body.getTerminator()).getValues()[0];
  Operation *combine = result.getDefiningOp();
  if (!combine || combine->getName().getStringRef() != name ||
      combine->getNumOperands() != 2 ||
      combine->getOperand(0) != body.getArgument(0) ||
      combine->getOperand(1) != body.getArgument(1))
    return {};
  return reduce;
}

void call(linalg::GenericOp op, PatternRewriter &rewriter, StringRef name,
          ArrayRef<Value> inputs, ArrayRef<uint32_t> scalars = {}) {
  Location loc = op.getLoc();
  auto type = cast<RankedTensorType>(op.getResult(0).getType());
  Value output = rewriter.create<memref::AllocOp>(
      loc, MemRefType::get(type.getShape(), type.getElementType()));
  SmallVector<Value> args;
  args.push_back(rewriter.create<memref::CastOp>(
      loc, UnrankedMemRefType::get(type.getElementType(), 0), output));
  for (Value input : inputs) {
    auto tensor = cast<RankedTensorType>(input.getType());
    auto layout = StridedLayoutAttr::get(
        rewriter.getContext(), ShapedType::kDynamic,
        SmallVector<int64_t>(tensor.getRank(), ShapedType::kDynamic));
    auto buffer = rewriter.create<bufferization::ToBufferOp>(
        loc,
        MemRefType::get(tensor.getShape(), tensor.getElementType(), layout),
        input, rewriter.getUnitAttr());
    args.push_back(rewriter.create<memref::CastOp>(
        loc, UnrankedMemRefType::get(tensor.getElementType(), 0), buffer));
  }
  for (uint32_t bits : scalars)
    args.push_back(rewriter.create<arith::ConstantIntOp>(loc, bits, 32));
  auto module = op->getParentOfType<ModuleOp>();
  if (!module.lookupSymbol<func::FuncOp>(name)) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    SmallVector<Type> types;
    for (Value arg : args)
      types.push_back(arg.getType());
    auto fn = rewriter.create<func::FuncOp>(
        loc, name, rewriter.getFunctionType(types, TypeRange{}));
    fn.setPrivate();
    fn->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
  }
  rewriter.create<func::CallOp>(loc, name, TypeRange{}, args);
  Value tensor = rewriter.create<bufferization::ToTensorOp>(
      loc, type, output, rewriter.getUnitAttr(), rewriter.getUnitAttr());
  rewriter.replaceOp(op, tensor);
}

struct Norm : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &r) const override {
    if (op.getNumResults() != 1 || op.getLibraryCallAttr())
      return failure();
    auto type = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!type || !type.hasStaticShape() || !type.getElementType().isF32() ||
        type.getRank() < 1)
      return failure();
    auto final = operation(op.getResult(0), "arith.mulf");
    if (!final)
      return failure();
    for (unsigned weighted = 0; weighted < 2; ++weighted) {
      Value weight = source(final->getOperand(weighted));
      auto wt = dyn_cast<RankedTensorType>(weight.getType());
      auto scaled = operation(final->getOperand(1 - weighted), "arith.mulf");
      if (!wt || wt.getRank() != 1 ||
          wt.getShape()[0] != type.getShape().back() || !scaled)
        continue;
      for (unsigned scale = 0; scale < 2; ++scale) {
        Value input;
        Value origin = source(scaled->getOperand(1 - scale));
        for (Value candidate : op.getInputs())
          if (candidate.getType() == type && source(candidate) == origin)
            input = candidate;
        auto rsqrt = operation(scaled->getOperand(scale), "math.rsqrt");
        if (!rsqrt || !input)
          continue;
        auto add = operation(rsqrt->getOperand(0), "arith.addf");
        if (!add)
          continue;
        auto eps = floatBits(add->getOperand(1));
        auto mean = operation(add->getOperand(0), "arith.mulf");
        if (!eps || !mean)
          continue;
        auto multiplier = floatBits(mean->getOperand(1));
        auto sum = reduction(mean->getOperand(0), "arith.addf", 0.0);
        if (!multiplier || !sum)
          continue;
        auto square = operation(sum.getInputs()[0], "math.fpowi");
        bool squares = square && constant(square->getOperand(1), 2.0) &&
                       source(square->getOperand(0)) == source(input);
        if (auto mul = operation(sum.getInputs()[0], "arith.mulf"))
          squares |= source(mul->getOperand(0)) == source(input) &&
                     source(mul->getOperand(1)) == source(input);
        if (!squares)
          continue;
        call(op, r, "rvv_norm", {input, weight}, {*multiplier, *eps});
        return success();
      }
    }
    return failure();
  }
};

struct Softmax : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &r) const override {
    if (op.getNumResults() != 1 || op.getLibraryCallAttr())
      return failure();
    auto type = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!type || !type.hasStaticShape() || !type.getElementType().isF32() ||
        type.getRank() < 1)
      return failure();
    auto mul = operation(op.getResult(0), "arith.mulf");
    if (!mul)
      return failure();
    for (unsigned side = 0; side < 2; ++side) {
      Value exps = source(mul->getOperand(side));
      auto reciprocal = operation(mul->getOperand(1 - side), "arith.divf");
      if (!reciprocal || !constant(reciprocal->getOperand(0), 1.0))
        continue;
      auto sum = reduction(reciprocal->getOperand(1), "arith.addf", 0.0);
      auto exp = operation(exps, "math.exp");
      if (!sum || sum.getInputs()[0] != exps || !exp)
        continue;
      auto sub = operation(exp->getOperand(0), "arith.subf");
      if (!sub)
        continue;
      Value input = source(sub->getOperand(0));
      auto max = reduction(sub->getOperand(1), "arith.maximumf",
                           -double(std::numeric_limits<float>::max()));
      if (!max || max.getInputs()[0] != input || input.getType() != type)
        continue;
      if (auto select = operation(input, "arith.select")) {
        auto scale = operation(select->getOperand(2), "arith.mulf");
        auto preprocessing = input.getDefiningOp<linalg::GenericOp>();
        auto maskArg = dyn_cast<BlockArgument>(select->getOperand(0));
        if (!preprocessing || !maskArg ||
            maskArg.getOwner() != preprocessing.getBody() ||
            maskArg.getArgNumber() >= preprocessing.getNumDpsInputs())
          continue;
        Value mask = preprocessing.getInputs()[maskArg.getArgNumber()];
        auto maskType = dyn_cast<RankedTensorType>(mask.getType());
        auto maskedBits = floatBits(select->getOperand(1));
        if (scale && maskedBits && type.getRank() == 4 &&
            type.getShape()[0] == 1 && maskType &&
            maskType.getElementType().isInteger(1) &&
            maskType.getShape() == ArrayRef<int64_t>{1, 1, type.getShape()[2],
                                                     type.getShape()[3]}) {
          for (unsigned side = 0; side < 2; ++side) {
            auto scaleBits = floatBits(scale->getOperand(side));
            auto scoresArg =
                dyn_cast<BlockArgument>(scale->getOperand(1 - side));
            if (!scoresArg || scoresArg.getOwner() != preprocessing.getBody() ||
                scoresArg.getArgNumber() >= preprocessing.getNumDpsInputs())
              continue;
            Value scores = preprocessing.getInputs()[scoresArg.getArgNumber()];
            if (scaleBits && scores.getType() == type) {
              bool mapsMatch = true;
              for (unsigned arg :
                   {maskArg.getArgNumber(), scoresArg.getArgNumber()}) {
                auto map = preprocessing.getIndexingMapsArray()[arg];
                auto shape = cast<RankedTensorType>(
                                 preprocessing.getInputs()[arg].getType())
                                 .getShape();
                if (map.getNumDims() != 4 || map.getNumSymbols() != 0 ||
                    map.getNumResults() != 4) {
                  mapsMatch = false;
                  break;
                }
                for (auto [axis, expr] : llvm::enumerate(map.getResults())) {
                  if (auto dim = dyn_cast<AffineDimExpr>(expr))
                    mapsMatch &= dim.getPosition() == axis;
                  else if (auto zero = dyn_cast<AffineConstantExpr>(expr))
                    mapsMatch &= zero.getValue() == 0 && shape[axis] == 1;
                  else
                    mapsMatch = false;
                }
              }
              if (!mapsMatch)
                continue;
              call(op, r, "rvv_attention_softmax", {scores, mask},
                   {*scaleBits, *maskedBits});
              return success();
            }
          }
        }
      }
      call(op, r, "rvv_softmax", {input});
      return success();
    }
    return failure();
  }
};

bool halfSlice(ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes,
               ArrayRef<int64_t> strides, RankedTensorType type,
               int64_t offset, unsigned axis = 3) {
  if (offsets.size() != 4 || sizes.size() != 4 || strides.size() != 4)
    return false;
  for (unsigned i = 0; i < 4; ++i)
    if (offsets[i] != (i == axis ? offset : 0) || strides[i] != 1 ||
        sizes[i] !=
            (i == axis ? type.getShape()[axis] / 2 : type.getShape()[i]))
      return false;
  return true;
}

struct Rope : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &r) const override {
    if (op.getNumResults() != 1 || op.getLibraryCallAttr())
      return failure();
    auto type = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!type || !type.hasStaticShape() || !type.getElementType().isF32() ||
        type.getRank() != 4 || type.getShape()[0] != 1 ||
        type.getShape()[3] % 2)
      return failure();
    auto add = operation(op.getResult(0), "arith.addf");
    if (!add)
      return failure();
    auto direct = operation(add->getOperand(0), "arith.mulf");
    auto rotated = operation(add->getOperand(1), "arith.mulf");
    if (!direct || !rotated)
      return failure();
    Value input;
    for (Value candidate : op.getInputs())
      if (candidate.getType() == type &&
          source(candidate) == source(direct->getOperand(0)))
        input = candidate;
    auto cos = operation(direct->getOperand(1), "math.cos");
    auto sin = operation(rotated->getOperand(1), "math.sin");
    if (!cos || !sin || !input ||
        source(cos->getOperand(0)) != source(sin->getOperand(0)))
      return failure();
    Value rotation;
    for (Value candidate : op.getInputs())
      if (source(candidate) == source(rotated->getOperand(0)))
        rotation = candidate;
    unsigned axis = 3;
    if (auto transpose = rotation.getDefiningOp<linalg::TransposeOp>()) {
      if (transpose.getPermutation() != ArrayRef<int64_t>{0, 2, 3, 1})
        return failure();
      rotation = transpose.getInput();
      axis = 1;
    }
    auto lowerInsert = rotation.getDefiningOp<tensor::InsertSliceOp>();
    if (!lowerInsert)
      return failure();
    auto concatType = cast<RankedTensorType>(lowerInsert.getType());
    if (!halfSlice(lowerInsert.getStaticOffsets(), lowerInsert.getStaticSizes(),
                   lowerInsert.getStaticStrides(), concatType,
                   type.getShape()[3] / 2, axis))
      return failure();
    Value lowerValue = lowerInsert.getSource();
    auto upperInsert =
        lowerInsert.getDest().getDefiningOp<tensor::InsertSliceOp>();
    if (!upperInsert)
      return failure();
    Value upperValue = upperInsert.getSource();
    if (axis == 1) {
      auto lowTranspose = lowerValue.getDefiningOp<linalg::TransposeOp>();
      auto highTranspose = upperValue.getDefiningOp<linalg::TransposeOp>();
      if (!lowTranspose || !highTranspose ||
          lowTranspose.getPermutation() != ArrayRef<int64_t>{0, 3, 1, 2} ||
          highTranspose.getPermutation() != ArrayRef<int64_t>{0, 3, 1, 2})
        return failure();
      lowerValue = lowTranspose.getInput();
      upperValue = highTranspose.getInput();
    }
    auto lower = lowerValue.getDefiningOp<tensor::ExtractSliceOp>();
    if (!lower || source(lower.getSource()) != source(input) ||
        !halfSlice(lower.getStaticOffsets(), lower.getStaticSizes(),
                   lower.getStaticStrides(), type, 0) ||
        !halfSlice(upperInsert.getStaticOffsets(), upperInsert.getStaticSizes(),
                   upperInsert.getStaticStrides(), concatType, 0, axis))
      return failure();
    auto negative = operation(upperValue, "arith.negf");
    if (!negative)
      return failure();
    auto upper =
        source(negative->getOperand(0)).getDefiningOp<tensor::ExtractSliceOp>();
    if (!upper || source(upper.getSource()) != source(input) ||
        !halfSlice(upper.getStaticOffsets(), upper.getStaticSizes(),
                   upper.getStaticStrides(), type, type.getShape()[3] / 2))
      return failure();
    auto collapse =
        source(cos->getOperand(0)).getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapse || collapse.getReassociationIndices() !=
                         SmallVector<ReassociationIndices>{{0}, {1, 2}})
      return failure();
    auto repeated = collapse.getSrc().getDefiningOp<linalg::GenericOp>();
    auto repeatType = dyn_cast<RankedTensorType>(collapse.getSrc().getType());
    if (!repeated || !repeatType ||
        repeatType.getShape() !=
            ArrayRef<int64_t>{type.getShape()[2], 2, type.getShape()[3] / 2})
      return failure();
    auto zeroAdd = operation(repeated.getResult(0), "arith.addf");
    if (!zeroAdd || floatBits(zeroAdd->getOperand(1)) != 0u)
      return failure();
    auto angle = operation(zeroAdd->getOperand(0), "arith.mulf");
    if (!angle)
      return failure();
    auto convert = operation(angle->getOperand(0), "arith.sitofp");
    Value freq = source(angle->getOperand(1));
    if (!convert ||
        freq.getType() !=
            RankedTensorType::get({type.getShape()[3] / 2}, r.getF32Type()))
      return failure();
    Value positions = source(convert->getOperand(0));
    if (positions.getType() !=
        RankedTensorType::get({type.getShape()[2]}, r.getI64Type()))
      return failure();
    call(op, r, "rvv_rope", {input, freq, positions});
    return success();
  }
};
} // namespace
void populateTensorPatterns(RewritePatternSet &patterns) {
  patterns.add<Norm, Softmax, Rope>(patterns.getContext());
}
} // namespace mlir::buddy::rvv

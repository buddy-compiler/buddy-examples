#include "RVVTensorKernels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
using namespace mlir;
namespace mlir::buddy::rvv {
namespace {
std::optional<uint32_t> bits(Value value) {
  auto op = expression(value).getDefiningOp<arith::ConstantOp>();
  if (!op)
    return std::nullopt;
  if (auto f = dyn_cast<FloatAttr>(op.getValue()); f && f.getType().isF32())
    return f.getValue().bitcastToAPInt().getZExtValue();
  return std::nullopt;
}
linalg::ReduceOp sum(Value value) {
  auto op = source(value).getDefiningOp<linalg::ReduceOp>();
  if (!op || op.getNumResults() != 1 || op.getInputs().size() != 1 ||
      op.getDimensions().size() != 1)
    return {};
  auto type = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
  if (!type || op.getDimensions()[0] != type.getRank() - 1)
    return {};
  auto fill = op.getInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fill || bits(fill.getInputs()[0]) != 0u)
    return {};
  auto &body = op.getRegion().front();
  auto yield = cast<linalg::YieldOp>(body.getTerminator()).getValues()[0];
  auto add = yield.getDefiningOp<arith::AddFOp>();
  if (!add || add.getLhs() != body.getArgument(0) ||
      add.getRhs() != body.getArgument(1))
    return {};
  return op;
}
Value call(linalg::GenericOp op, PatternRewriter &r, StringRef name,
           ValueRange inputs, ArrayRef<uint32_t> scalars = {}) {
  auto type = cast<RankedTensorType>(op.getResult(0).getType());
  auto loc = op.getLoc();
  auto unranked = UnrankedMemRefType::get(r.getF32Type(), 0);
  Value output = r.create<memref::AllocOp>(
      loc, MemRefType::get(type.getShape(), r.getF32Type()));
  SmallVector<Value> args{r.create<memref::CastOp>(loc, unranked, output)};
  for (auto input : inputs) {
    auto tensor = cast<RankedTensorType>(input.getType());
    auto layout = StridedLayoutAttr::get(
        r.getContext(), ShapedType::kDynamic,
        SmallVector<int64_t>(tensor.getRank(), ShapedType::kDynamic));
    Value buffer = r.create<bufferization::ToBufferOp>(
        loc,
        MemRefType::get(tensor.getShape(), tensor.getElementType(), layout),
        input, r.getUnitAttr());
    args.push_back(r.create<memref::CastOp>(loc, unranked, buffer));
  }
  for (auto scalar : scalars)
    args.push_back(r.create<arith::ConstantIntOp>(loc, scalar, 32));
  auto module = op->getParentOfType<ModuleOp>();
  if (!module.lookupSymbol<func::FuncOp>(name)) {
    OpBuilder::InsertionGuard guard(r);
    r.setInsertionPointToStart(module.getBody());
    SmallVector<Type> types;
    for (auto arg : args)
      types.push_back(arg.getType());
    auto fn = r.create<func::FuncOp>(loc, name,
                                     r.getFunctionType(types, TypeRange{}));
    fn.setPrivate();
    fn->setAttr("llvm.emit_c_interface", r.getUnitAttr());
  }
  r.create<func::CallOp>(loc, name, TypeRange{}, args);
  return r.create<bufferization::ToTensorOp>(loc, type, output, r.getUnitAttr(),
                                             r.getUnitAttr());
}
struct LayerNorm : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &r) const override {
    if (op.getNumResults() != 1 || op.getLibraryCallAttr())
      return failure();
    auto type = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!type || !type.hasStaticShape() || !type.getElementType().isF32() ||
        type.getRank() < 1)
      return failure();
    auto affine = operation(op.getResult(0), "arith.addf");
    auto weighted = operation(affine ? affine->getOperand(0) : op.getResult(0),
                              "arith.mulf");
    Value bias = affine ? source(affine->getOperand(1)) : Value{};
    if (!weighted)
      return failure();
    auto normalized = operation(weighted->getOperand(0), "arith.mulf");
    auto weight = source(weighted->getOperand(1));
    auto coefficient =
        RankedTensorType::get({type.getShape().back()}, r.getF32Type());
    auto biasType =
        bias ? dyn_cast<RankedTensorType>(bias.getType()) : RankedTensorType{};
    if (!normalized ||
        (bias && (!biasType || !biasType.hasStaticShape() ||
                  biasType.getNumElements() != type.getShape().back())) ||
        weight.getType() != coefficient)
      return failure();
    auto centered = operation(normalized->getOperand(0), "arith.subf");
    auto rsqrt = operation(normalized->getOperand(1), "math.rsqrt");
    if (!centered || !rsqrt)
      return failure();
    auto input = source(centered->getOperand(0));
    auto mean = operation(centered->getOperand(1), "arith.mulf");
    auto biased = operation(rsqrt->getOperand(0), "arith.addf");
    if (input.getType() != type || !mean || !biased)
      return failure();
    auto variance = operation(biased->getOperand(0), "arith.mulf");
    auto eps = bits(biased->getOperand(1));
    auto multiplier = bits(mean->getOperand(1));
    auto first = sum(mean->getOperand(0));
    if (!variance || !eps || !multiplier || !first ||
        source(first.getInputs()[0]) != input ||
        bits(variance->getOperand(1)) != multiplier)
      return failure();
    auto second = sum(variance->getOperand(0));
    if (!second)
      return failure();
    auto square = operation(second.getInputs()[0], "arith.mulf");
    if (!square)
      return failure();
    for (auto operand : square->getOperands()) {
      auto sub = operation(operand, "arith.subf");
      if (!sub || source(sub->getOperand(0)) != input ||
          operation(sub->getOperand(1), "arith.mulf") != mean)
        return failure();
    }
    if (bias)
      r.replaceOp(op,
                  call(op, r, "rvv_layernorm", ValueRange{input, weight, bias},
                       {*multiplier, *eps}));
    else
      r.replaceOp(op, call(op, r, "rvv_layernorm_no_bias",
                           ValueRange{input, weight}, {*multiplier, *eps}));
    return success();
  }
};

Value cloneExpression(Value value, Block *oldBlock, OpBuilder &builder,
                      IRMapping &map) {
  if (map.contains(value))
    return map.lookup(value);
  Operation *def = value.getDefiningOp();
  if (!def || def->getBlock() != oldBlock)
    return value;
  for (Value operand : def->getOperands())
    cloneExpression(operand, oldBlock, builder, map);
  builder.clone(*def, map);
  return map.lookup(value);
}
Value materialize(linalg::GenericOp op, Value scalar, PatternRewriter &r) {
  auto type = cast<RankedTensorType>(op.getResult(0).getType());
  for (Value input : op.getInputs())
    if (input.getType() == type && source(input) == source(scalar))
      return input;
  auto empty = r.create<tensor::EmptyOp>(op.getLoc(), type.getShape(),
                                         type.getElementType());
  auto prefix = r.create<linalg::GenericOp>(
      op.getLoc(), TypeRange{type}, op.getInputs(), ValueRange{empty},
      op.getIndexingMapsArray(), op.getIteratorTypesArray(),
      [&](OpBuilder &b, Location loc, ValueRange args) {
        IRMapping map;
        for (auto [old, newArg] : llvm::zip(op.getBody()->getArguments(), args))
          map.map(old, newArg);
        Value value = cloneExpression(scalar, op.getBody(), b, map);
        b.create<linalg::YieldOp>(loc, value);
      });
  return prefix.getResult(0);
}
struct Gelu : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &r) const override {
    if (op.getNumResults() != 1 || op.getLibraryCallAttr())
      return failure();
    auto type = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!type || !type.hasStaticShape() || !type.getElementType().isF32())
      return failure();
    Value gelu, input;
    for (auto &candidate : op.getBody()->without_terminator()) {
      if (candidate.getName().getStringRef() != "arith.mulf")
        continue;
      auto half = operation(candidate.getOperand(0), "arith.mulf");
      auto shift = operation(candidate.getOperand(1), "arith.addf");
      if (!half || !shift || !constant(half->getOperand(1), 0.5) ||
          !constant(shift->getOperand(1), 1.0))
        continue;
      auto erf = operation(shift->getOperand(0), "math.erf");
      if (!erf)
        continue;
      auto scaled = operation(erf->getOperand(0), "arith.mulf");
      if (!scaled ||
          !constant(scaled->getOperand(1), double(float(0.707106769))) ||
          expression(scaled->getOperand(0)) != expression(half->getOperand(0)))
        continue;
      gelu = candidate.getResult(0);
      input = half->getOperand(0);
      break;
    }
    if (!gelu)
      return failure();
    Value activation = materialize(op, input, r);
    Value result = call(op, r, "rvv_gelu", ValueRange{activation});
    auto yielded =
        cast<linalg::YieldOp>(op.getBody()->getTerminator()).getValues()[0];
    if (yielded == gelu) {
      r.replaceOp(op, result);
      return success();
    }
    SmallVector<Value> inputs(op.getInputs());
    inputs.push_back(result);
    SmallVector<AffineMap> maps(op.getIndexingMapsArray());
    maps.insert(maps.end() - 1, AffineMap::getMultiDimIdentityMap(
                                    type.getRank(), r.getContext()));
    auto empty = r.create<tensor::EmptyOp>(op.getLoc(), type.getShape(),
                                           type.getElementType());
    auto suffix = r.create<linalg::GenericOp>(
        op.getLoc(), TypeRange{type}, inputs, ValueRange{empty}, maps,
        op.getIteratorTypesArray(),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          IRMapping map;
          for (unsigned i = 0; i < op.getNumDpsInputs(); ++i)
            map.map(op.getBody()->getArgument(i), args[i]);
          map.map(op.getBody()->getArgument(op.getNumDpsInputs()), args.back());
          map.map(gelu, args[op.getNumDpsInputs()]);
          b.create<linalg::YieldOp>(
              loc, cloneExpression(yielded, op.getBody(), b, map));
        });
    r.replaceOp(op, suffix.getResults());
    return success();
  }
};
} // namespace
void populateNormalizationPatterns(RewritePatternSet &patterns) {
  patterns.add<LayerNorm, Gelu>(patterns.getContext());
}
} // namespace mlir::buddy::rvv

//===- HostShapes.cpp - Reshapes the GPU lowering can take ----------------===//
//
// gvn-host-shapes, on a bufferized module for the GPU. The GPU lowering
// (convert-memcpy-to-gpu) moves heap allocations and globals into device
// memory, where the host can read nothing, but leaves the stack alone:
// - a static allocation the host alone uses (loads, stores, the shape of a
//   reshape, the bookkeeping of the deallocation helper) is allocated on the
//   stack;
// - every memref.reshape (from tensor.reshape) becomes a
//   memref.reinterpret_cast of its contiguous source, with the sizes of its
//   shape and contiguous strides: a shape held by a constant global gives
//   static sizes, a shape built at run time is read on the stack.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
class HostShapesPass
    : public PassWrapper<HostShapesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HostShapesPass)
  StringRef getArgument() const final { return "gvn-host-shapes"; }
  StringRef getDescription() const final {
    return "Reinterpret contiguous buffers for reshapes, their shapes read on "
           "the host";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect>();
  }

  // Whether the host alone uses `value`, a buffer.
  static bool hostOnly(Value value) {
    return llvm::all_of(value.getUses(), [](OpOperand &use) {
      Operation *user = use.getOwner();
      if (auto cast = dyn_cast<memref::CastOp>(user))
        return hostOnly(cast.getResult());
      if (auto reshape = dyn_cast<memref::ReshapeOp>(user))
        return &use == &reshape.getShapeMutable();
      if (auto call = dyn_cast<func::CallOp>(user))
        return call.getCallee() == "dealloc_helper";
      return isa<memref::LoadOp, memref::StoreOp, memref::DeallocOp>(user);
    });
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<memref::AllocOp> host;
    module.walk([&](memref::AllocOp allocation) {
      if (allocation.getType().hasStaticShape() &&
          hostOnly(allocation.getResult()))
        host.push_back(allocation);
    });
    for (memref::AllocOp allocation : host) {
      OpBuilder builder(allocation);
      Value stack = memref::AllocaOp::create(builder, allocation.getLoc(),
                                             allocation.getType());
      for (Operation *user : llvm::make_early_inc_range(allocation->getUsers()))
        if (isa<memref::DeallocOp>(user))
          user->erase();
      allocation.getResult().replaceAllUsesWith(stack);
      allocation.erase();
    }

    SmallVector<memref::ReshapeOp> reshapes;
    module.walk(
        [&](memref::ReshapeOp reshape) { reshapes.push_back(reshape); });
    for (memref::ReshapeOp reshape : reshapes) {
      OpBuilder builder(reshape);
      Location loc = reshape.getLoc();
      Value shape = reshape.getShape();
      // The constant shape of a global.
      DenseIntElementsAttr constant;
      if (auto global = shape.getDefiningOp<memref::GetGlobalOp>()) {
        auto definition =
            module.lookupSymbol<memref::GlobalOp>(global.getNameAttr());
        constant = dyn_cast_or_null<DenseIntElementsAttr>(
            definition.getConstantInitValue());
        if (!constant)
          llvm::report_fatal_error("gvn-host-shapes: a shape global without "
                                   "a constant value");
      } else if (!reshape.getType().hasStaticShape() &&
                 !shape.getDefiningOp<memref::AllocaOp>()) {
        reshape.emitOpError("requires a constant global or host stack shape: ")
            << shape;
        signalPassFailure();
        return;
      }

      auto result = reshape.getType();
      SmallVector<OpFoldResult> sizes, strides(result.getRank());
      for (int64_t dim = 0; dim < result.getRank(); ++dim) {
        if (!result.isDynamicDim(dim)) {
          sizes.push_back(builder.getIndexAttr(result.getDimSize(dim)));
        } else if (constant) {
          sizes.push_back(builder.getIndexAttr(
              constant.getValues<APInt>()[dim].getSExtValue()));
        } else {
          sizes.push_back(
              memref::LoadOp::create(
                  builder, loc, shape,
                  ValueRange{arith::ConstantIndexOp::create(builder, loc, dim)})
                  .getResult());
        }
      }
      OpFoldResult stride = builder.getIndexAttr(1);
      for (int64_t dim = result.getRank() - 1; dim >= 0; --dim) {
        strides[dim] = stride;
        auto multiply = [&](OpFoldResult a, OpFoldResult b) -> OpFoldResult {
          auto value = [&](OpFoldResult v) -> Value {
            if (auto attribute = dyn_cast<Attribute>(v))
              return arith::ConstantIndexOp::create(
                  builder, loc, cast<IntegerAttr>(attribute).getInt());
            return cast<Value>(v);
          };
          if (isa<Attribute>(a) && isa<Attribute>(b))
            return builder.getIndexAttr(
                cast<IntegerAttr>(cast<Attribute>(a)).getInt() *
                cast<IntegerAttr>(cast<Attribute>(b)).getInt());
          return arith::MulIOp::create(builder, loc, value(a), value(b))
              .getResult();
        };
        stride = multiply(stride, sizes[dim]);
      }
      SmallVector<int64_t> staticStrides;
      for (OpFoldResult each : strides)
        staticStrides.push_back(
            isa<Attribute>(each)
                ? cast<IntegerAttr>(cast<Attribute>(each)).getInt()
                : ShapedType::kDynamic);
      auto contiguous = MemRefType::get(
          result.getShape(), result.getElementType(),
          StridedLayoutAttr::get(module.getContext(), 0, staticStrides));
      Value view = memref::ReinterpretCastOp::create(
          builder, loc, contiguous, reshape.getSource(),
          builder.getIndexAttr(0), sizes, strides);
      reshape.getResult().replaceAllUsesWith(
          memref::CastOp::create(builder, loc, result, view).getResult());
      reshape.erase();
    }
  }
};
} // namespace

namespace gvn {
void registerHostShapesPass() { PassRegistration<HostShapesPass>(); }
} // namespace gvn

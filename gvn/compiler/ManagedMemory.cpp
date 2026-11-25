//===- ManagedMemory.cpp - Put every buffer in managed memory -------------===//
//
// gvn-managed-memory, on a module whose kernels are outlined and asynchronous
// (gpu-async-region) for the GPU: every heap allocation becomes a host-shared
// gpu.alloc (CUDA managed memory, allocated synchronously), its deallocation
// a gpu.dealloc, every copy a gpu.memcpy, and every global read a copy of it
// in managed memory; the copies and deallocations wait for the GPU before and
// after them. Host and GPU then read and write every buffer, wherever it was
// computed: the graphs of a device program call one another through the
// program, on the host, and the program's caller passes managed buffers too.
// Stack allocations (gvn-host-shapes) stay: the host alone uses them.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
class ManagedMemoryPass
    : public PassWrapper<ManagedMemoryPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ManagedMemoryPass)
  StringRef getArgument() const final { return "gvn-managed-memory"; }
  StringRef getDescription() const final {
    return "Allocate every buffer in managed memory";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
  }

  // A token once the GPU is done.
  static Value wait(OpBuilder &builder, Location loc) {
    return gpu::WaitOp::create(builder, loc,
                               gpu::AsyncTokenType::get(builder.getContext()),
                               ValueRange())
        .getAsyncToken();
  }

  // Copies `source` into `target` with the GPU idle before and after.
  static Operation *memcpy(OpBuilder &builder, Location loc, Value target,
                           Value source) {
    Value token = wait(builder, loc);
    auto copy = gpu::MemcpyOp::create(builder, loc, token.getType(),
                                      ValueRange{token}, target, source);
    gpu::WaitOp::create(builder, loc, Type(), ValueRange{copy.getAsyncToken()});
    return copy;
  }

  static MemRefType identity(MemRefType type) {
    return MemRefType::get(type.getShape(), type.getElementType());
  }

  void runOnOperation() override {
    SmallVector<Operation *> ops;
    getOperation().walk([&](Operation *op) {
      if (isa<gpu::AllocOp, memref::AllocOp, memref::DeallocOp, memref::CopyOp,
              memref::GetGlobalOp>(op))
        ops.push_back(op);
    });
    for (Operation *op : ops) {
      OpBuilder builder(op);
      Location loc = op->getLoc();
      if (auto allocation = dyn_cast<gpu::AllocOp>(op)) {
        if (!allocation.getAsyncDependencies().empty())
          gpu::WaitOp::create(builder, loc, Type(),
                              allocation.getAsyncDependencies());
        auto managed = gpu::AllocOp::create(
            builder, loc, allocation.getMemref().getType(), Type(),
            ValueRange(), allocation.getDynamicSizes(),
            allocation.getSymbolOperands(),
            /*hostShared=*/true);
        allocation.getMemref().replaceAllUsesWith(managed.getMemref());
        if (Value token = allocation.getAsyncToken())
          token.replaceAllUsesWith(wait(builder, loc));
        allocation.erase();
      } else if (auto allocation = dyn_cast<memref::AllocOp>(op)) {
        auto managed = gpu::AllocOp::create(
            builder, loc, allocation.getType(), Type(), ValueRange(),
            allocation.getDynamicSizes(), allocation.getSymbolOperands(),
            /*hostShared=*/true);
        allocation.getResult().replaceAllUsesWith(managed.getMemref());
        allocation.erase();
      } else if (auto deallocation = dyn_cast<memref::DeallocOp>(op)) {
        Value token = wait(builder, loc);
        token =
            gpu::DeallocOp::create(builder, loc, token.getType(),
                                   ValueRange{token}, deallocation.getMemref())
                .getAsyncToken();
        gpu::WaitOp::create(builder, loc, Type(), ValueRange{token});
        deallocation.erase();
      } else if (auto copy = dyn_cast<memref::CopyOp>(op)) {
        auto source = cast<MemRefType>(copy.getSource().getType());
        auto target = cast<MemRefType>(copy.getTarget().getType());
        if ((!source.getLayout().isIdentity() &&
             (!memref::isStaticShapeAndContiguousRowMajor(source) ||
              source.getStridesAndOffset().second != 0)) ||
            (!target.getLayout().isIdentity() &&
             (!memref::isStaticShapeAndContiguousRowMajor(target) ||
              target.getStridesAndOffset().second != 0)))
          llvm::report_fatal_error("gvn-managed-memory: a strided copy "
                                   "(convert-strided-memref-copy-to-linalg "
                                   "first)");
        Value from = copy.getSource(), to = copy.getTarget();
        if (!source.getLayout().isIdentity() ||
            !target.getLayout().isIdentity()) {
          auto flat = MemRefType::get({source.getNumElements()},
                                      source.getElementType());
          from = memref::ReinterpretCastOp::create(
              builder, loc, flat, from, builder.getIndexAttr(0),
              ArrayRef<OpFoldResult>{
                  builder.getIndexAttr(source.getNumElements())},
              ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
        }
        if (!source.getLayout().isIdentity() ||
            !target.getLayout().isIdentity()) {
          auto flat = MemRefType::get({target.getNumElements()},
                                      target.getElementType());
          to = memref::ReinterpretCastOp::create(
              builder, loc, flat, to, builder.getIndexAttr(0),
              ArrayRef<OpFoldResult>{
                  builder.getIndexAttr(target.getNumElements())},
              ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
        }
        memcpy(builder, loc, to, from);
        copy.erase();
      } else {
        auto global = cast<memref::GetGlobalOp>(op);
        builder.setInsertionPointAfter(global);
        auto type = identity(global.getType());
        auto managed = gpu::AllocOp::create(builder, loc, type, Type(),
                                            ValueRange(), ValueRange(),
                                            ValueRange(), /*hostShared=*/true);
        Operation *filled =
            memcpy(builder, loc, managed.getMemref(), global.getResult());
        global.getResult().replaceAllUsesExcept(managed.getMemref(), filled);
      }
    }
  }
};
} // namespace

namespace gvn {
void registerManagedMemoryPass() { PassRegistration<ManagedMemoryPass>(); }
} // namespace gvn

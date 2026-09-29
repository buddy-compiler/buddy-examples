#include "Conversion/LowerBuckyball/LowerBuckyball.h"
#include "Target/BuckyballTargetRegistry.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
class RuntimeLowering
    : public PassWrapper<RuntimeLowering, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RuntimeLowering)
  StringRef getArgument() const final { return "lower-tile-runtime"; }
  StringRef getDescription() const final {
    return "Dispatch compiled subgraphs through the tile runtime";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    for (auto [oldName, newName] : {std::pair{"malloc", "workspace_alloc"},
                                    std::pair{"free", "workspace_free"}}) {
      if (auto function = module.lookupSymbol<LLVM::LLVMFuncOp>(oldName)) {
        if (failed(SymbolTable::replaceAllSymbolUses(
                function, StringAttr::get(context, newName), module))) {
          function.emitError("cannot bind workspace allocator");
          return signalPassFailure();
        }
        function.setSymName(newName);
        if (StringRef(newName) == "workspace_alloc")
          function.setResultAttr(0, "llvm.noalias", UnitAttr::get(context));
      }
    }
    SmallVector<LLVM::CallOp> calls;
    module.walk([&](LLVM::CallOp call) {
      auto callee = call.getCallee();
      if (!callee || (!callee->starts_with("subgraph") &&
                      !callee->starts_with("_mlir_ciface_subgraph")))
        return;
      auto function = module.lookupSymbol<LLVM::LLVMFuncOp>(*callee);
      if (function && function.isExternal())
        calls.push_back(call);
    });
    if (calls.empty())
      return;

    OpBuilder b(context);
    auto pointer = LLVM::LLVMPointerType::get(context);
    auto voidType = LLVM::LLVMVoidType::get(context);
    b.setInsertionPointToStart(module.getBody());
    auto runtime = b.create<LLVM::LLVMFuncOp>(
        module.getLoc(), "task_run",
        LLVM::LLVMFunctionType::get(voidType,
                                    {b.getI64Type(), pointer, pointer}));
    unsigned index = 0;
    for (LLVM::CallOp call : calls) {
      Location loc = call.getLoc();
      auto function = module.lookupSymbol<LLVM::LLVMFuncOp>(*call.getCallee());
      auto targetName = function->getAttrOfType<StringAttr>("buckyball.target");
      const auto &target = targetName
                               ? buckyball_target::getBuckyballTarget(targetName.getValue())
                               : buckyball_target::getBuckyballTarget();
      SmallVector<Type> fields(call.getOperandTypes());
      if (call.getNumResults())
        fields.push_back(call.getResult().getType());
      auto frameType = LLVM::LLVMStructType::getLiteral(context, fields);
      b.setInsertionPointToEnd(module.getBody());
      auto callback = b.create<LLVM::LLVMFuncOp>(
          loc, "invoke_" + std::to_string(index++),
          LLVM::LLVMFunctionType::get(voidType, {pointer}),
          LLVM::Linkage::Internal);
      Block *entry = callback.addEntryBlock(b);
      b.setInsertionPointToStart(entry);
      IRMapping mapping;
      for (auto [field, operand] : llvm::enumerate(call.getOperands())) {
        Value address = b.create<LLVM::GEPOp>(
            loc, pointer, frameType, entry->getArgument(0),
            ArrayRef<LLVM::GEPArg>{0, int32_t(field)});
        mapping.map(operand,
                    b.create<LLVM::LoadOp>(loc, operand.getType(), address));
      }
      auto invocation = cast<LLVM::CallOp>(b.clone(*call, mapping));
      if (call.getNumResults()) {
        Value result = b.create<LLVM::GEPOp>(
            loc, pointer, frameType, entry->getArgument(0),
            ArrayRef<LLVM::GEPArg>{0, int32_t(fields.size() - 1)});
        b.create<LLVM::StoreOp>(loc, invocation.getResult(), result);
      }
      b.create<LLVM::ReturnOp>(loc, ValueRange{});

      b.setInsertionPoint(call);
      Value one = b.create<LLVM::ConstantOp>(loc, b.getI64Type(),
                                             b.getI64IntegerAttr(1));
      Value frame = b.create<LLVM::AllocaOp>(loc, pointer, frameType, one, 16);
      for (auto [field, operand] : llvm::enumerate(call.getOperands())) {
        Value address =
            b.create<LLVM::GEPOp>(loc, pointer, frameType, frame,
                                  ArrayRef<LLVM::GEPArg>{0, int32_t(field)});
        b.create<LLVM::StoreOp>(loc, operand, address);
      }
      Value kind = b.create<LLVM::ConstantOp>(loc, b.getI64Type(),
                                              b.getI64IntegerAttr(target.signature));
      Value code =
          b.create<LLVM::AddressOfOp>(loc, pointer, callback.getSymName());
      b.create<LLVM::CallOp>(loc, runtime, ValueRange{kind, code, frame});
      if (call.getNumResults()) {
        Value result = b.create<LLVM::GEPOp>(
            loc, pointer, frameType, frame,
            ArrayRef<LLVM::GEPArg>{0, int32_t(fields.size() - 1)});
        call.getResult().replaceAllUsesWith(
            b.create<LLVM::LoadOp>(loc, call.getResult().getType(), result));
      }
      call.erase();
    }
  }
};
} // namespace

void mlir::buddy::registerTileRuntimePass() {
  PassRegistration<RuntimeLowering>();
}

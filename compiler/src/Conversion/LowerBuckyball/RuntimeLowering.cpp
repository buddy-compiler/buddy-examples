#include "Conversion/LowerBuckyball/LowerBuckyball.h"
#include "Target/BuckyballTargetRegistry.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSwitch.h"

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
    if (module->getAttrOfType<UnitAttr>("buckyball.local"))
      return;
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
    LLVM::LLVMFuncOp placedRuntime;
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
      if (auto core = function->getAttrOfType<IntegerAttr>("buckyball.core")) {
        if (core.getInt() < 1 || core.getInt() > 255) {
          function.emitError("logical compute core must be in [1, 255]");
          return signalPassFailure();
        }
        if (!placedRuntime) {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(module.getBody());
          placedRuntime = b.create<LLVM::LLVMFuncOp>(module.getLoc(), "task_run_on",
              LLVM::LLVMFunctionType::get(voidType, {b.getI64Type(), b.getI64Type(), pointer, pointer}));
        }
        Value destination = b.create<LLVM::ConstantOp>(loc, b.getI64Type(), b.getI64IntegerAttr(core.getInt()));
        b.create<LLVM::CallOp>(loc, placedRuntime, ValueRange{destination, kind, code, frame});
      } else {
        b.create<LLVM::CallOp>(loc, runtime, ValueRange{kind, code, frame});
      }
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

class AntRuntimeLowering
    : public PassWrapper<AntRuntimeLowering, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AntRuntimeLowering)
  StringRef getArgument() const final { return "lower-ant-runtime"; }
  StringRef getDescription() const final {
    return "Submit lowered NPU instructions from the controller to Ant";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> instructions;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef().starts_with("buckyball.intr."))
        instructions.push_back(op);
    });
    if (instructions.empty())
      return;
    OpBuilder builder(&getContext());
    auto type = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(&getContext()),
        {builder.getI32Type(), builder.getI64Type(), builder.getI64Type()});
    auto emit = module.lookupSymbol<LLVM::LLVMFuncOp>("ant_emit");
    if (emit && emit.getFunctionType() != type) {
      emit.emitError("ant_emit must have signature void(i32, i64, i64)");
      return signalPassFailure();
    }
    if (!emit) {
      builder.setInsertionPointToStart(module.getBody());
      emit = builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "ant_emit", type);
    }
    for (Operation *op : instructions) {
      if (op->getNumOperands() != 2 || op->getNumResults() != 0 ||
          !llvm::all_of(op->getOperandTypes(),
                        [](Type type) { return type.isInteger(64); })) {
        op->emitError("Ant submission requires two i64 operands and no result");
        return signalPassFailure();
      }
      StringRef name = op->getName().getStringRef().drop_front(15);
      int32_t funct7;
      if (name == "custom") {
        funct7 = cast<IntegerAttr>(op->getAttr("funct7")).getInt();
      } else {
        funct7 = llvm::StringSwitch<int32_t>(name)
                     .Case("fence", 0)
                     .Case("mvout", 16)
                     .Case("mset", 32)
                     .Case("mvin", 33)
                     .Case("mvin_mmio", 35)
                     .Default(-1);
      }
      if (funct7 < 0 || funct7 > 127) {
        op->emitError("unsupported Ant command encoding");
        return signalPassFailure();
      }
      builder.setInsertionPoint(op);
      Value code = builder.create<LLVM::ConstantOp>(
          op->getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(funct7));
      builder.create<LLVM::CallOp>(op->getLoc(), emit,
                                  ValueRange{code, op->getOperand(0), op->getOperand(1)});
      op->erase();
    }
  }
};
} // namespace

void mlir::buddy::registerTileRuntimePass() {
  PassRegistration<RuntimeLowering>();
  PassRegistration<AntRuntimeLowering>();
}

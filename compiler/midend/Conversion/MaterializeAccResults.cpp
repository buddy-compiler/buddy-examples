#include "Acc/AccOps.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace ::buddy::acc;

namespace {
// Export CPU-produced results through the hart's coherent DMA path. A host
// simulator can then inspect the destination in DDR without reading CPU caches.
class MaterializeAccResultsPass
    : public PassWrapper<MaterializeAccResultsPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeAccResultsPass)
  StringRef getArgument() const final { return "materialize-acc-results"; }
  StringRef getDescription() const final {
    return "Export final tensor results through coherent accelerator DMA";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<AccDialect, arith::ArithDialect,
                bufferization::BufferizationDialect, memref::MemRefDialect>();
  }
  void runOnOperation() override {
    auto function = getOperation();
    if (function.isExternal())
      return;
    bool computes = false;
    auto rootBuffer = [](Value value) {
      while (auto view = value.getDefiningOp<ViewLikeOpInterface>())
        value = view.getViewSource();
      return value;
    };
    llvm::DenseSet<Value> dmaOutputs;
    function.walk([&](Operation *op) {
      StringRef dialect = op->getName().getDialectNamespace();
      computes |= dialect == "bbacc" || dialect.ends_with("_ball");
      if (op->getName().getStringRef().starts_with("bbacc.bank_mvout"))
        dmaOutputs.insert(rootBuffer(op->getOperand(0)));
    });
    if (!computes)
      return;
    auto *terminator = function.front().getTerminator();
    OpBuilder b(terminator);
    for (OpOperand &returned : terminator->getOpOperands()) {
      Value value = returned.get();
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type)
        continue;
      if (auto tensor = value.getDefiningOp<bufferization::ToTensorOp>())
        if (dmaOutputs.contains(rootBuffer(tensor.getBuffer())))
          continue;
      if (!type.hasStaticShape() || type.getRank() == 0) {
        function.emitOpError(
            "result materialization requires static ranked tensors");
        return signalPassFailure();
      }
      Location loc = value.getLoc();
      int64_t elementBytes = type.getElementTypeBitWidth() / 8;
      int64_t rowBytes = target::getTarget().bankWidthBits / 8;
      if (elementBytes <= 0 || rowBytes % elementBytes) {
        function.emitOpError("result elements must divide the bank row width");
        return signalPassFailure();
      }
      int64_t lanes = rowBytes / elementBytes;
      int64_t capacity =
          std::min(target::getTarget().bankDepth,
                   (target::getTarget().dmaMaxRanges - 1) * 4096 / rowBytes) *
          lanes;
      if (capacity <= 0) {
        function.emitOpError("DMA range capacity cannot hold a bank row");
        return signalPassFailure();
      }
      auto memoryType = MemRefType::get(type.getShape(), type.getElementType());
      Value source =
          bufferization::ToBufferOp::create(b, loc, memoryType, value, true);
      Value output = memref::AllocOp::create(b, loc, memoryType);
      ReassociationIndices axes;
      for (int64_t axis = 0; axis < type.getRank(); ++axis)
        axes.push_back(axis);
      Value flatInput = source, flatOutput = output;
      if (type.getRank() > 1) {
        flatInput = memref::CollapseShapeOp::create(
            b, loc, source, ArrayRef<ReassociationIndices>{axes});
        flatOutput = memref::CollapseShapeOp::create(
            b, loc, output, ArrayRef<ReassociationIndices>{axes});
      }
      auto index = [&](int64_t x) -> Value {
        return arith::ConstantIndexOp::create(b, loc, x);
      };
      for (int64_t first = 0; first < type.getNumElements();
           first += capacity) {
        int64_t count = std::min(capacity, type.getNumElements() - first);
        Value bank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
        for (int direction = 0; direction < 2; ++direction) {
          Value memory = direction ? flatOutput : flatInput;
          for (int part = 0; part < 2; ++part) {
            int64_t length = part ? count % lanes : count / lanes * lanes;
            if (!length)
              continue;
            int64_t offset = part ? count / lanes * lanes : 0;
            Value view = memref::SubViewOp::create(
                b, loc, memory,
                ArrayRef<OpFoldResult>{b.getIndexAttr(first + offset)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(length)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
            Value row = index(offset / lanes),
                  rows = index(part ? 1 : count / lanes);
            Value bytes = index(part ? length * elementBytes : rowBytes);
            if (direction)
              bank = BankMvoutStridedOp::create(
                  b, loc, b.getI64Type(), view, bank, row, rows,
                  index(rowBytes), 0, 1, bytes, index(0));
            else
              bank = BankMvinStridedOp::create(b, loc, b.getI64Type(), view,
                                               bank, row, rows, index(rowBytes),
                                               0, 1, bytes, index(0));
          }
        }
        BankReleaseOp::create(b, loc, bank);
      }
      returned.set(
          bufferization::ToTensorOp::create(b, loc, type, output, true, true));
    }
  }
};
} // namespace
namespace mlir::buddy {
void registerMaterializeAccResultsPass() {
  PassRegistration<MaterializeAccResultsPass>();
}
} // namespace mlir::buddy

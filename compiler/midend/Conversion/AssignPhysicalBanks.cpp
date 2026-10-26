//===- AssignPhysicalBanks.cpp - Map virtual banks to the target banks ----===//
//
// Walks each function in program order. A bank_alloc takes the lowest free
// bank and row x col of the physical banks of the target; bank_release frees
// them. The virtual operations become their physical operations on that bank.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
class AssignPhysicalBanksPass
    : public PassWrapper<AssignPhysicalBanksPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssignPhysicalBanksPass)

  StringRef getArgument() const final { return "assign-physical-banks"; }
  StringRef getDescription() const final {
    return "Map the virtual banks to the banks of the target";
  }

  void runOnOperation() override {
    const buckyball_target::BuckyballTargetConfig &target =
        buckyball_target::getBuckyballTarget();
    // The physical banks each bank holds; 0 when the bank is free.
    SmallVector<int64_t> held(target.privateBankMax + 1, 0);
    int64_t physical = 0;
    DenseMap<Value, int64_t> banks;
    SmallVector<Operation *> virtualOps;

    auto bankOf = [&](Operation *op, Value handle) -> FailureOr<int64_t> {
      auto bank = banks.find(handle);
      if (bank == banks.end())
        return op->emitOpError("uses a bank not allocated before it");
      return bank->second;
    };

    WalkResult result = getOperation().walk([&](Operation *op) {
      OpBuilder builder(op);
      Location loc = op->getLoc();
      if (auto alloc = dyn_cast<BankAllocOp>(op)) {
        int64_t size = alloc.getRow() * alloc.getCol();
        int64_t *free = llvm::find(held, 0);
        if (free == held.end() || physical + size > target.bankNum) {
          alloc.emitOpError("finds no free bank of ")
              << alloc.getRow() << "x" << alloc.getCol();
          return WalkResult::interrupt();
        }
        *free = size;
        physical += size;
        int64_t bank = free - held.begin();
        banks[alloc.getBank()] = bank;
        MsetOp::create(builder, loc, bank, true, alloc.getRow(),
                       alloc.getCol());
      } else if (auto release = dyn_cast<BankReleaseOp>(op)) {
        FailureOr<int64_t> bank = bankOf(op, release.getBank());
        if (failed(bank))
          return WalkResult::interrupt();
        physical -= held[*bank];
        held[*bank] = 0;
        MsetOp::create(builder, loc, *bank, false, 0, 0);
      } else if (auto mvin = dyn_cast<BankMvinOp>(op)) {
        FailureOr<int64_t> bank = bankOf(op, mvin.getBank());
        if (failed(bank))
          return WalkResult::interrupt();
        banks[mvin.getBankOut()] = *bank;
        MvinOp::create(builder, loc, mvin.getInput(), *bank, mvin.getDepth(),
                       mvin.getStride());
      } else if (auto mvout = dyn_cast<BankMvoutOp>(op)) {
        FailureOr<int64_t> bank = bankOf(op, mvout.getBank());
        if (failed(bank))
          return WalkResult::interrupt();
        banks[mvout.getBankOut()] = *bank;
        MvoutOp::create(builder, loc, mvout.getOutput(), *bank,
                        mvout.getDepth(), mvout.getStride());
      } else if (auto custom = dyn_cast<BankCustomOp>(op)) {
        SmallVector<int64_t> sources;
        for (Value source : custom.getSources()) {
          FailureOr<int64_t> bank = bankOf(op, source);
          if (failed(bank))
            return WalkResult::interrupt();
          sources.push_back(*bank);
        }
        FailureOr<int64_t> destination = bankOf(op, custom.getDestination());
        if (failed(destination))
          return WalkResult::interrupt();
        banks[custom.getDestinationOut()] = *destination;
        auto physicalOp = CustomOp::create(
            builder, loc, custom.getArguments(),
            builder.getDenseI64ArrayAttr(sources),
            builder.getI64IntegerAttr(*destination), custom.getKindAttr(),
            custom.getReadDestinationAttr());
        // The other attributes belong to the computation of the kind.
        physicalOp->setDiscardableAttrs(custom->getDiscardableAttrDictionary());
      } else {
        return WalkResult::advance();
      }
      virtualOps.push_back(op);
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();
    if (physical != 0) {
      getOperation().emitOpError("returns with banks not released");
      return signalPassFailure();
    }
    // Users come after their banks: erase them first.
    for (Operation *op : llvm::reverse(virtualOps))
      op->erase();
  }
};
} // namespace

namespace mlir::buddy {
void registerAssignPhysicalBanksPass() {
  PassRegistration<AssignPhysicalBanksPass>();
}
} // namespace mlir::buddy

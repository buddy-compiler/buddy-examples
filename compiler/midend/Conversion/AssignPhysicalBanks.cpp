//===- AssignPhysicalBanks.cpp - Map virtual banks to the target banks ----===//
//
// Walks each function in program order. A bank_alloc takes the lowest free
// bank but bank 0, the zero bank, and row x col of the physical banks of the
// target; bank_release frees them; bank_transfer moves the groups of a bank to
// another. The virtual operations become their physical operations on that
// bank; a Ball operation builds its own through BankSSALowering. A bank
// carried through an scf.for is the same bank in every iteration.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"
#include "Interfaces/BankInterfaces.h"
#include "Target/BuckyballTargetRegistry.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
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

    // Pre-order: a loop maps its carried banks before its body uses them.
    WalkResult result =
        getOperation().walk<WalkOrder::PreOrder>([&](Operation *op) {
          OpBuilder builder(op);
          Location loc = op->getLoc();
          if (auto loop = dyn_cast<scf::ForOp>(op)) {
            for (auto [init, carried, result] :
                 llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs(),
                           loop.getResults())) {
              if (!banks.contains(init))
                continue;
              int64_t bank = banks.lookup(init);
              banks[carried] = bank;
              banks[result] = bank;
            }
            return WalkResult::advance();
          }
          if (auto yield = dyn_cast<scf::YieldOp>(op)) {
            auto loop = dyn_cast<scf::ForOp>(yield->getParentOp());
            if (!loop)
              return WalkResult::advance();
            for (auto [value, carried] :
                 llvm::zip(yield.getOperands(), loop.getRegionIterArgs())) {
              auto bank = banks.find(carried);
              if (bank == banks.end())
                continue;
              FailureOr<int64_t> yielded = bankOf(op, value);
              if (failed(yielded))
                return WalkResult::interrupt();
              if (*yielded != bank->second) {
                yield.emitOpError(
                    "carries a different bank to the next iteration");
                return WalkResult::interrupt();
              }
            }
            return WalkResult::advance();
          }
          if (auto alloc = dyn_cast<BankAllocOp>(op)) {
            int64_t size = alloc.getRow() * alloc.getCol();
            int64_t *free = std::find(held.begin() + 1, held.end(), 0);
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
          } else if (auto transfer = dyn_cast<BankTransferOp>(op)) {
            FailureOr<int64_t> source = bankOf(op, transfer.getSource());
            FailureOr<int64_t> target = bankOf(op, transfer.getTarget());
            if (failed(source) || failed(target))
              return WalkResult::interrupt();
            held[*target] += held[*source];
            held[*source] = 0;
            banks[transfer.getTargetOut()] = *target;
            MsetTransferOp::create(builder, loc, *source, *target);
          } else if (auto mvin = dyn_cast<BankMvinOp>(op)) {
            FailureOr<int64_t> bank = bankOf(op, mvin.getBank());
            if (failed(bank))
              return WalkResult::interrupt();
            banks[mvin.getBankOut()] = *bank;
            MvinOp::create(builder, loc, mvin.getInput(), *bank,
                           mvin.getDepth(), mvin.getStride());
          } else if (auto mvout = dyn_cast<BankMvoutOp>(op)) {
            FailureOr<int64_t> bank = bankOf(op, mvout.getBank());
            if (failed(bank))
              return WalkResult::interrupt();
            banks[mvout.getBankOut()] = *bank;
            MvoutOp::create(builder, loc, mvout.getOutput(), *bank,
                            mvout.getDepth(), mvout.getStride());
          } else if (auto ball = dyn_cast<BankSSALowering>(op)) {
            SmallVector<int64_t> sources;
            for (Value source : ball.getSourceBanks()) {
              FailureOr<int64_t> bank = bankOf(op, source);
              if (failed(bank))
                return WalkResult::interrupt();
              sources.push_back(*bank);
            }
            FailureOr<int64_t> destination =
                bankOf(op, ball.getDestinationBank());
            if (failed(destination))
              return WalkResult::interrupt();
            banks[op->getResult(0)] = *destination;
            ball.buildPhysical(builder, sources, *destination);
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
    // Users come after their banks: erase them first. A loop still carrying a
    // bank gets its physical number instead.
    for (Operation *op : llvm::reverse(virtualOps)) {
      for (Value handle : op->getResults()) {
        if (handle.use_empty())
          continue;
        OpBuilder builder(op);
        handle.replaceAllUsesWith(arith::ConstantIntOp::create(
            builder, op->getLoc(), banks.lookup(handle), 64));
      }
      op->erase();
    }
  }
};
} // namespace

namespace mlir::buddy {
void registerAssignPhysicalBanksPass() {
  PassRegistration<AssignPhysicalBanksPass>();
}
} // namespace mlir::buddy

//===- AssignPhysicalBanks.cpp - Map virtual banks to the target banks ----===//
//
// Walks each function in program order. A bank_alloc takes the lowest free
// bank but bank 0, the zero bank, and row x col of the physical banks of the
// target; bank_release frees them; bank_transfer moves the groups of a bank to
// another. The virtual operations (with bank_mvin_group and bank_run_operator
// of the GP domain) become their physical operations on that bank; a Ball
// operation builds its own through BankSSALowering. A bank carried through an
// scf.for is the same bank in every iteration.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccDialect.h"
#include "Acc/AccOps.h"
#include "Acc/BankInterfaces.h"
#include "Conversion/Buckyball.h"
#include "Target/TargetRegistry.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace ::buddy::acc;

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
    const target::TargetConfig &target = target::getTarget();
    if (auto profile =
            getOperation()->getAttrOfType<StringAttr>("bbacc.target");
        profile && profile.getValue() != target.hart)
      return;
    // The physical banks each bank holds; 0 when the bank is free.
    SmallVector<int64_t> held(target.privateBankMax + 1, 0);
    int64_t physical = 0;
    DenseMap<Value, int64_t> banks;
    SmallVector<Operation *> virtualOps;
    // The physical banks held when each loop starts.
    DenseMap<Operation *, int64_t> entry;
    // The geometry of the bank each release frees.
    DenseMap<Operation *, std::pair<int64_t, int64_t>> freed;
    DenseMap<int64_t, std::pair<int64_t, int64_t>> geometry;

    auto bankOf = [&](Operation *op, Value handle) -> FailureOr<int64_t> {
      auto bank = banks.find(handle);
      if (bank == banks.end())
        return op->emitOpError("uses a bank not allocated before it");
      return bank->second;
    };

    // Pre-order: a loop maps its carried banks before its body uses them.
    WalkResult result = getOperation().walk<WalkOrder::PreOrder>([&](Operation
                                                                         *op) {
      OpBuilder builder(op);
      Location loc = op->getLoc();
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        entry[loop] = physical;
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
        if (auto branch = dyn_cast<scf::IfOp>(yield->getParentOp());
            branch && !branch.getElseRegion().empty() &&
            yield->getBlock() == &branch.getElseRegion().front()) {
          auto thenYield = cast<scf::YieldOp>(
              branch.getThenRegion().front().getTerminator());
          for (auto [result, left, right] :
               llvm::zip(branch.getResults(), thenYield.getOperands(),
                         yield.getOperands())) {
            if (!banks.contains(left) && !banks.contains(right))
              continue;
            auto a = bankOf(op, left), b = bankOf(op, right);
            if (failed(a) || failed(b))
              return WalkResult::interrupt();
            if (*a != *b) {
              yield.emitOpError("branches yield different physical banks");
              return WalkResult::interrupt();
            }
            banks[result] = *a;
          }
          return WalkResult::advance();
        }
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
            yield.emitOpError("carries a different bank to the next iteration");
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
        geometry[bank] = {alloc.getRow(), alloc.getCol()};
        MsetOp::create(builder, loc, bank, true, alloc.getRow(),
                       alloc.getCol());
      } else if (auto clear = dyn_cast<BankClearOp>(op)) {
        FailureOr<int64_t> bank = bankOf(op, clear.getBank());
        if (failed(bank))
          return WalkResult::interrupt();
        if (held[*bank] != clear.getRow() * clear.getCol()) {
          clear.emitOpError("clear geometry differs from allocation");
          return WalkResult::interrupt();
        }
        banks[clear.getBankOut()] = *bank;
        ClearOp::create(builder, loc, *bank, clear.getRow(), clear.getCol());
      } else if (auto release = dyn_cast<BankReleaseOp>(op)) {
        FailureOr<int64_t> bank = bankOf(op, release.getBank());
        if (failed(bank))
          return WalkResult::interrupt();
        physical -= held[*bank];
        held[*bank] = 0;
        freed[MsetOp::create(builder, loc, *bank, false, 0, 0)] =
            geometry.lookup(*bank);
      } else if (auto transfer = dyn_cast<BankTransferOp>(op)) {
        FailureOr<int64_t> source = bankOf(op, transfer.getSource());
        FailureOr<int64_t> target = bankOf(op, transfer.getTarget());
        if (failed(source) || failed(target))
          return WalkResult::interrupt();
        held[*target] += held[*source];
        held[*source] = 0;
        // The groups of a transfer leave the geometry of no alloc.
        geometry[*target] = {-1, -1};
        banks[transfer.getTargetOut()] = *target;
        MsetTransferOp::create(builder, loc, *source, *target);
      } else if (auto mvin = dyn_cast<BankMvinOp>(op)) {
        FailureOr<int64_t> bank = bankOf(op, mvin.getBank());
        if (failed(bank))
          return WalkResult::interrupt();
        banks[mvin.getBankOut()] = *bank;
        MvinOp::create(builder, loc, mvin.getInput(), *bank, mvin.getDepth(),
                       mvin.getStride());
      } else if (auto mvin = dyn_cast<BankMvinGroupOp>(op)) {
        FailureOr<int64_t> bank = bankOf(op, mvin.getBank());
        if (failed(bank))
          return WalkResult::interrupt();
        banks[mvin.getBankOut()] = *bank;
        MvinGroupOp::create(builder, loc, mvin.getInput(), *bank,
                            mvin.getGroup(), mvin.getDepth(), mvin.getStride());
      } else if (auto dma = dyn_cast<BankMvinStridedOp>(op)) {
        FailureOr<int64_t> bank = bankOf(op, dma.getBank());
        if (failed(bank))
          return WalkResult::interrupt();
        banks[dma.getBankOut()] = *bank;
        MvinStridedOp::create(
            builder, loc, dma.getMemory(), *bank, dma.getFirst(),
            dma.getDepth(), dma.getMemoryStride(), dma.getGroup(),
            dma.getBankStride(), dma.getBytes(), dma.getBlockLog2());
      } else if (auto dma = dyn_cast<BankMvoutStridedOp>(op)) {
        FailureOr<int64_t> bank = bankOf(op, dma.getBank());
        if (failed(bank))
          return WalkResult::interrupt();
        banks[dma.getBankOut()] = *bank;
        MvoutStridedOp::create(
            builder, loc, dma.getMemory(), *bank, dma.getFirst(),
            dma.getDepth(), dma.getMemoryStride(), dma.getGroup(),
            dma.getBankStride(), dma.getBytes(), dma.getBlockLog2());
      } else if (auto run = dyn_cast<BankRunOperatorOp>(op)) {
        FailureOr<int64_t> input = bankOf(op, run.getInput());
        FailureOr<int64_t> output = bankOf(op, run.getOutput());
        if (failed(input) || failed(output))
          return WalkResult::interrupt();
        banks[run.getOutputOut()] = *output;
        RunOperatorOp::create(builder, loc, *input, *output, run.getProgram());
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
        FailureOr<int64_t> destination = bankOf(op, ball.getDestinationBank());
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
    // Bank loop arguments are architectural handles, already assigned above.
    // Materialize them before canonicalization removes their now-unused
    // carries.
    getOperation().walk([&](scf::ForOp loop) {
      for (auto [carried, result] :
           llvm::zip(loop.getRegionIterArgs(), loop.getResults())) {
        auto bank = banks.find(carried);
        if (bank == banks.end())
          continue;
        OpBuilder builder(loop);
        result.replaceAllUsesWith(arith::ConstantIntOp::create(
            builder, loop.getLoc(), bank->second, 64));
        builder.setInsertionPointToStart(loop.getBody());
        carried.replaceAllUsesWith(arith::ConstantIntOp::create(
            builder, loop.getLoc(), bank->second, 64));
      }
    });
    // A release followed by an alloc of the same bank and geometry in the same
    // block, with no loop or transfer between, keeps the bank instead.
    auto merge = [&](Block &block) {
      SmallVector<std::pair<Operation *, Operation *>> kept;
      for (MsetOp release : block.getOps<MsetOp>()) {
        if (release.getAlloc() || !freed.contains(release))
          continue;
        for (Operation *op = release->getNextNode(); op;
             op = op->getNextNode()) {
          if (op->getNumRegions() || isa<MsetTransferOp>(op))
            break;
          auto mset = dyn_cast<MsetOp>(op);
          if (!mset || mset.getBank() != release.getBank())
            continue;
          if (mset.getAlloc() &&
              std::pair<int64_t, int64_t>(mset.getRow(), mset.getCol()) ==
                  freed.lookup(release))
            kept.push_back({release, mset});
          break;
        }
      }
      for (auto [release, alloc] : kept) {
        release->erase();
        alloc->erase();
      }
    };
    // A bank allocated and released once in every iteration is held through
    // the loop instead: the bank alias table renames each write of the next
    // iteration, so its loads overlap the work of this one. A plain mset alloc
    // leaves the contents undefined, so no iteration reads what the last left.
    getOperation().walk([&](scf::ForOp loop) {
      Block &body = *loop.getBody();
      merge(body);
      DenseMap<int64_t, int64_t> allocs, releases;
      DenseSet<int64_t> transferred;
      loop.getBody()->walk([&](Operation *op) {
        if (auto mset = dyn_cast<MsetOp>(op))
          ++(mset.getAlloc() ? allocs : releases)[mset.getBank()];
        if (auto transfer = dyn_cast<MsetTransferOp>(op)) {
          transferred.insert(transfer.getSource());
          transferred.insert(transfer.getTarget());
        }
      });
      SmallVector<MsetOp> hoisted;
      int64_t start = entry.lookup(loop);
      for (MsetOp mset : body.getOps<MsetOp>()) {
        int64_t bank = mset.getBank();
        if (!mset.getAlloc() || allocs[bank] != 1 || releases[bank] != 1 ||
            transferred.contains(bank))
          continue;
        auto release = llvm::find_if(body.getOps<MsetOp>(), [&](MsetOp op) {
          return op.getBank() == bank && !op.getAlloc();
        });
        if (release == body.getOps<MsetOp>().end())
          continue;
        hoisted.push_back(mset);
        hoisted.push_back(*release);
        start += mset.getRow() * mset.getCol();
      }
      // The hoisted banks stay held while the others come and go.
      DenseMap<int64_t, int64_t> size;
      int64_t held = start;
      bool fits = held <= target.bankNum;
      loop.getBody()->walk([&](Operation *op) {
        if (auto transfer = dyn_cast<MsetTransferOp>(op)) {
          size[transfer.getTarget()] += size[transfer.getSource()];
          size[transfer.getSource()] = 0;
        }
        auto mset = dyn_cast<MsetOp>(op);
        if (!mset || llvm::is_contained(hoisted, mset))
          return;
        if (mset.getAlloc()) {
          size[mset.getBank()] = mset.getRow() * mset.getCol();
          held += size[mset.getBank()];
        } else {
          held -= size[mset.getBank()];
        }
        fits &= held <= target.bankNum;
      });
      if (!fits)
        return;
      for (MsetOp mset : hoisted) {
        if (mset.getAlloc())
          mset->moveBefore(loop);
        else
          mset->moveAfter(loop);
      }
      entry[loop] = start;
    });
    getOperation().walk([&](Block *block) { merge(*block); });
  }
};
} // namespace

namespace mlir::buddy {
void registerAssignPhysicalBanksPass() {
  PassRegistration<AssignPhysicalBanksPass>();
}
} // namespace mlir::buddy

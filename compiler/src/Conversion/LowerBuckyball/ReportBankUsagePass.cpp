//===- ReportBankUsagePass.cpp - Report physical bank usage ---------------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>

using namespace mlir;

namespace {

class ReportBankUsagePass
    : public PassWrapper<ReportBankUsagePass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReportBankUsagePass)
  ReportBankUsagePass() = default;
  ReportBankUsagePass(const ReportBankUsagePass &) {}

  StringRef getArgument() const final { return "report-bank-usage"; }
  StringRef getDescription() const final {
    return "Report mapped group occupancy against the supplied pool capacity "
           "from mset alloc/transfer/release timeline.";
  }

  Option<int64_t> bankNum{*this, "bank_num",
                          llvm::cl::desc("Physical group capacity of the reported pool."),
                          llvm::cl::init(16)};
  Option<bool> verbose{*this, "verbose",
                       llvm::cl::desc("Print per-event timeline."),
                       llvm::cl::init(false)};

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (bankNum <= 0) {
      func.emitError("report-bank-usage: bank_num must be > 0");
      signalPassFailure();
      return;
    }

    llvm::DenseMap<int64_t, int64_t> allocSize;
    int64_t cur = 0;
    int64_t peak = 0;
    int64_t allocCnt = 0;
    int64_t relCnt = 0;
    int64_t evt = 0;

    auto getConstI64 = [&](Value v) -> std::optional<int64_t> {
      auto c = v.getDefiningOp<arith::ConstantOp>();
      if (!c)
        return std::nullopt;
      auto ai = dyn_cast<IntegerAttr>(c.getValue());
      if (!ai)
        return std::nullopt;
      return ai.getInt();
    };

    for (Block &blk : func.getBlocks()) {
      for (Operation &op : blk.getOperations()) {
        if (auto transfer = dyn_cast<::buddy::buckyball::MsetTransferOp>(op)) {
          ++evt;
          auto source = getConstI64(transfer.getSource());
          auto target = getConstI64(transfer.getTarget());
          if (!source || !target) {
            transfer.emitError(
                "report-bank-usage: transfer bank IDs must be constant");
            signalPassFailure();
            return;
          }
          auto sourceAllocation = allocSize.find(*source);
          if (sourceAllocation == allocSize.end()) {
            transfer.emitError(
                "report-bank-usage: transfer source is not allocated");
            signalPassFailure();
            return;
          }
          int64_t groups = sourceAllocation->second;
          allocSize.erase(sourceAllocation);
          allocSize[*target] += groups;
          if (verbose)
            llvm::errs() << "[bank-usage] " << func.getName() << " evt=" << evt
                         << " transfer b" << *source << " -> b" << *target
                         << " groups=" << groups << " cur=" << cur << "/"
                         << bankNum << "\n";
          continue;
        }
        auto mset = dyn_cast<::buddy::buckyball::MsetOp>(op);
        if (!mset)
          continue;
        ++evt;
        auto bid = getConstI64(mset.getBankId());
        if (!bid) {
          func.emitError("report-bank-usage: mset bank id must be constant");
          signalPassFailure();
          return;
        }
        if (*bid < 0 || *bid > 1023) {
          func.emitError("report-bank-usage: bank id out of range");
          signalPassFailure();
          return;
        }

        if (mset.getAlloc()) {
          int64_t row = mset.getRow();
          int64_t col = mset.getCol();
          int64_t need = col == 0 ? static_cast<int64_t>(bankNum) : col;
          if (row < 0 || row > 31 || col < 0 || col > 31 ||
              cur + need > bankNum) {
            func.emitError(
                "report-bank-usage: invalid allocation or capacity exceeded");
            signalPassFailure();
            return;
          }
          if (allocSize.count(*bid)) {
            func.emitError(
                "report-bank-usage: double alloc on same virtual bank");
            signalPassFailure();
            return;
          }
          allocSize[*bid] = need;
          cur += need;
          peak = std::max(peak, cur);
          ++allocCnt;
          if (verbose) {
            llvm::errs() << "[bank-usage] " << func.getName() << " evt=" << evt
                         << " alloc b" << *bid << " row=" << row
                         << " col=" << col << " cur=" << cur << "/" << bankNum
                         << "\n";
          }
          continue;
        }

        auto it = allocSize.find(*bid);
        if (it == allocSize.end()) {
          func.emitError("report-bank-usage: release without prior alloc");
          signalPassFailure();
          return;
        }
        int64_t need = it->second;
        allocSize.erase(it);
        cur -= need;
        ++relCnt;
        if (verbose) {
          llvm::errs() << "[bank-usage] " << func.getName() << " evt=" << evt
                       << " release b" << *bid << " size=" << need
                       << " cur=" << cur << "/" << bankNum << "\n";
        }
      }
    }

    llvm::errs() << "[bank-usage] " << func.getName() << " peak=" << peak << "/"
                 << bankNum << " alloc=" << allocCnt << " release=" << relCnt
                 << " leaked=" << allocSize.size() << "\n";

    if (!allocSize.empty()) {
      func.emitError("report-bank-usage: leaked allocations at function end");
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::buddy::registerReportBankUsagePass() {
  PassRegistration<ReportBankUsagePass>();
}

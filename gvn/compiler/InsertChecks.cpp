//===- InsertChecks.cpp - Mark the slice boundaries of a hart graph -------===//
//
// gvn-insert-checks, on the output of the hart's convert-linalg-to-acc
// compiled from a linalg input with its locations: in every function of hart
// computations (one with bank layer operations), every tensor a top-level
// bufferization.to_tensor produces from a DMA-written buffer
// gets a gvn.check after its producer and the cycle markers that observe it.
// The check names the linalg operation the value equals: the first location
// of the producer, `line:column` in the linalg input. A slice boundary holds
// nothing on device: no check goes where a bank lives across, a bank produced
// by a top-level operation of the bank layer (or holding one) before it and
// used after it; such a result is no boundary and its slice extends to the
// next. The slices of the function are the spans up to each check from the
// previous one. A returned tensor also gets a boundary so that CPU output
// assembly after the last bank computation is verified.
//
//===----------------------------------------------------------------------===//

#include "GvnOps.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace {
std::string checkId(Operation *producer) {
  Location loc = producer->getLoc();
  while (true) {
    if (auto fused = dyn_cast<FusedLoc>(loc))
      loc = fused.getLocations().front();
    else if (auto call = dyn_cast<CallSiteLoc>(loc))
      loc = call.getCallee();
    else
      break;
  }
  auto file = dyn_cast<FileLineColLoc>(loc);
  if (!file) {
    producer->emitOpError("has no linalg location for a gvn check");
    llvm::report_fatal_error("gvn-insert-checks: compile the linalg input "
                             "with its locations");
  }
  return std::to_string(file.getLine()) + ":" +
         std::to_string(file.getColumn());
}

// A value the compiler made, no value of the source graph: its location is
// a name with no source under it. GVN neither checks nor reads it.
bool generated(Location loc) {
  while (true) {
    if (auto fused = dyn_cast<FusedLoc>(loc))
      loc = fused.getLocations().front();
    else if (auto call = dyn_cast<CallSiteLoc>(loc))
      loc = call.getCallee();
    else
      break;
  }
  auto name = dyn_cast<NameLoc>(loc);
  return name && isa<UnknownLoc>(name.getChildLoc());
}

class SourceOriginsPass
    : public PassWrapper<SourceOriginsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SourceOriginsPass)
  StringRef getArgument() const final { return "gvn-source-origins"; }
  StringRef getDescription() const final {
    return "Retain original tensor operations and types through hart "
           "partitioning";
  }
  void runOnOperation() override {
    Builder builder(&getContext());
    getOperation().walk([&](Operation *op) {
      if (op->getNumResults() != 1 ||
          !isa<RankedTensorType>(op->getResult(0).getType()))
        return;
      NamedAttrList origin;
      origin.set("gvn.operation",
                 builder.getStringAttr(op->getName().getStringRef()));
      origin.set("gvn.type", TypeAttr::get(op->getResult(0).getType()));
      op->setLoc(FusedLoc::get(&getContext(), {op->getLoc()},
                               origin.getDictionary(&getContext())));
    });
  }
};

class InsertChecksPass
    : public PassWrapper<InsertChecksPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertChecksPass)
  StringRef getArgument() const final { return "gvn-insert-checks"; }
  StringRef getDescription() const final {
    return "Mark the DDR tensors of a hart graph as slice boundaries";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<::gvn::GvnDialect, memref::MemRefDialect,
                    bufferization::BufferizationDialect>();
  }
  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (function.isExternal())
      return;
    // Only the functions of the hart computations: those with bank layer
    // operations; the graphs that launch them on hardware threads have none.
    bool computes = false;
    function.walk([&](Operation *op) {
      StringRef dialect = op->getName().getDialectNamespace();
      computes |= dialect == "bbacc" || dialect.ends_with("_ball");
    });
    if (!computes)
      return;
    Block &body = function.front();
    DenseMap<Operation *, int64_t> position;
    for (Operation &op : body)
      position[&op] = position.size();

    // live[p]: the banks living across the point after operation p.
    SmallVector<int64_t> live(position.size() + 1, 0);
    for (Operation &op : body.without_terminator()) {
      bool banks = false;
      op.walk([&](Operation *nested) {
        StringRef dialect = nested->getName().getDialectNamespace();
        banks |= dialect == "bbacc" || dialect.ends_with("_ball");
      });
      if (!banks)
        continue;
      for (Value result : op.getResults()) {
        if (!result.getType().isInteger(64))
          continue;
        int64_t last = position[&op];
        for (Operation *user : result.getUsers())
          last = std::max(last, position[body.findAncestorOpInBlock(*user)]);
        ++live[position[&op]];
        --live[last];
      }
    }
    for (size_t point = 1; point < live.size(); ++point)
      live[point] += live[point - 1];

    auto rootBuffer = [](Value value) {
      while (auto view = value.getDefiningOp<ViewLikeOpInterface>())
        value = view.getViewSource();
      return value;
    };
    llvm::DenseSet<Value> dmaOutputs;
    function.walk([&](Operation *op) {
      if (op->getName().getStringRef().starts_with("bbacc.bank_mvout"))
        dmaOutputs.insert(rootBuffer(op->getOperand(0)));
    });
    SmallVector<Value> values;
    // A launched thread subgraph exposes its results to its parent. Keep its
    // fused stages together rather than inventing intermediate launch results.
    if (!function->hasAttr("bbacc.target"))
      for (Operation &op : body.without_terminator())
        if (auto tensor = dyn_cast<bufferization::ToTensorOp>(op))
          if (dmaOutputs.contains(rootBuffer(tensor.getBuffer())))
            values.push_back(op.getResult(0));
    // A trimmed result can be copied from DMA scratch into its destination.
    // Its returned tensor remains a boundary after the CPU copy completes.
    for (Value value : body.getTerminator()->getOperands())
      if (isa<RankedTensorType>(value.getType()) && value.getDefiningOp() &&
          !llvm::is_contained(values, value))
        values.push_back(value);
    OpBuilder builder(function.getContext());
    for (Value value : values) {
      if (llvm::any_of(value.getUsers(), [](Operation *user) {
            return isa<::gvn::CheckOp>(user);
          }))
        continue;
      Operation *producer = value.getDefiningOp();
      if (generated(producer->getLoc()))
        continue;
      Operation *after = producer;
      while (after->getNextNode()->getName().getStringRef() ==
                 "frontend.trace_end" &&
             after->getNextNode()->getOperand(0) == value)
        after = after->getNextNode();
      if (live[position[after]])
        continue;
      builder.setInsertionPointAfter(after);
      ::gvn::CheckOp::create(builder, producer->getLoc(), value,
                             checkId(producer));
    }
  }
};
// gvn-insert-op-checks{ops=<names>}, on a linalg input compiled with its
// locations, for an accelerator that takes whole linalg operations from DDR
// (Gemmini): every top-level operation named in ops gets a gvn.check after
// it. A slice is one such operation alone (gvn-slice-op): the work of the
// accelerator; the CPU work between them is no slice.
class InsertOpChecksPass
    : public PassWrapper<InsertOpChecksPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertOpChecksPass)
  InsertOpChecksPass() = default;
  InsertOpChecksPass(const InsertOpChecksPass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "gvn-insert-op-checks"; }
  StringRef getDescription() const final {
    return "Mark the results of the accelerator's linalg operations as slice "
           "boundaries";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<::gvn::GvnDialect>();
  }

  ListOption<std::string> ops{*this, "ops",
                              llvm::cl::desc("The accelerator's operations")};

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (function.isExternal())
      return;
    Block &body = function.front();
    SmallVector<Value> values;
    for (Operation &op : body.without_terminator())
      if (llvm::is_contained(ops, op.getName().getStringRef().str()))
        values.push_back(op.getResult(0));
    OpBuilder builder(function.getContext());
    for (Value value : values) {
      Operation *producer = value.getDefiningOp();
      builder.setInsertionPointAfter(producer);
      ::gvn::CheckOp::create(builder, producer->getLoc(), value,
                             checkId(producer));
    }
  }
};
} // namespace

namespace gvn {
void registerInsertChecksPass() {
  PassRegistration<SourceOriginsPass>();
  PassRegistration<InsertChecksPass>();
  PassRegistration<InsertOpChecksPass>();
}
} // namespace gvn

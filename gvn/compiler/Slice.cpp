//===- Slice.cpp - Cut one slice out of a checked hart graph --------------===//
//
// gvn-slice{function=<f> index=k layout=<json>}, on the output of
// gvn-insert-checks: the function f of hart computations computes its slice
// k only, the top-level operations after its check k-1 up to its check k.
//
// External values come from GPU input fixtures or BEMU DDR checkpoints through
// the byte array gvn_in; the external word array gvn_table gives, for
// each such value, its byte offset in gvn_in and its dynamic sizes, so that
// one compiled slice serves every instance of it (another token count,
// another cache length). The value of check k is written in place into the
// external byte array gvn_out: its buffer becomes a view of gvn_out, which
// the hart computation writes when it runs, after the CPU recorded it.
//
// Operations before the slice stay when the slice needs them: those defining
// a value it reads, those touching a buffer it touches (before the slice,
// not those only loading it into banks), and the start of every cycle trace
// the slice ends; nothing that computes a value of an earlier check through
// its tensor is kept, nor the start of a cycle trace a later slice ends. Every
// other operation is removed; a result the slice does not compute is returned
// uninitialized.
//
// With iterate=true, the slice is a bounded batch of tile iterations. The
// output writes of the span form units (units()): a static loop holding
// some, or a segment of operations no bank outlives, beside the setup banks
// held across all of them; a loop holding units runs one trip at a time,
// one holding none runs `batch` trips (the batch axis, which the driver
// chooses: a cycle budget per instance). The iteration, a row-major index
// over the units, is a word of gvn_table; a loop not selected runs no trip,
// a segment not selected is skipped by an scf.if. A batch never splits a
// tile iteration: no bank holds live data across its bounds. The tensors of
// CPU computations (linalg, frontend) the iteration reads come from gvn_in
// instead of being computed. The iteration writes its part of the value of
// check k; the harness initializes unwritten bytes and captures the BEMU
// result.
//
// When f is a subgraph launched on a hardware thread (hart.launch), its
// graph keeps that launch only (the first, when the layers launch it): the
// transfers into the thread stay, their sources that the graph computes come
// from gvn_in, and the results of the graph are uninitialized, sized by
// gvn_table. gvn_entry then calls the public function of the graph with
// arguments read from gvn_in. When the subgraph computes a partition of the
// checked value (the graph inserts it into the full value), gvn_out is the full
// value and the transfer writes the partition at its coordinates there; the
// full sizes are words of gvn_table.
//
// The layout file lists the entries of gvn_table: each value read from
// gvn_in (a check of f, a value of the graph, an argument of the public
// function) with its slot, the slots of the sizes of the graph results and,
// for a partition, of the full value in gvn_out (kind "output").
//
//===----------------------------------------------------------------------===//

#include "GvnOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <map>
#include <vector>

using namespace mlir;

namespace {
// The external arrays are declared with this many elements; the harness
// defines them with their real sizes.
constexpr int64_t externalSize = int64_t(1) << 31;

std::string elementName(Type type) {
  if (type.isF32())
    return "f32";
  if (type.isInteger(1))
    return "i1";
  if (type.isInteger(8))
    return "i8";
  if (type.isInteger(32))
    return "i32";
  if (type.isInteger(64))
    return "i64";
  llvm::report_fatal_error("gvn: unsupported element type of a slice value");
}

std::string typeName(Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return text;
}

// The buffer a memref views.
Value rootBuffer(Value value) {
  while (auto view = value.getDefiningOp<ViewLikeOpInterface>())
    value = view.getViewSource();
  return value;
}

// `line:column` of the first location of `op` in the linalg input.
std::string locationId(Location loc) {
  while (true) {
    if (auto fused = dyn_cast<FusedLoc>(loc))
      loc = fused.getLocations().front();
    else if (auto call = dyn_cast<CallSiteLoc>(loc))
      loc = call.getCallee();
    else
      break;
  }
  auto file = dyn_cast<FileLineColLoc>(loc);
  if (!file)
    llvm::report_fatal_error("gvn-slice: a value has no linalg location");
  return std::to_string(file.getLine()) + ":" +
         std::to_string(file.getColumn());
}

StringRef nameOf(Operation *op) { return op->getName().getStringRef(); }

FlatSymbolRefAttr calleeOf(Operation *launch) {
  return launch->getAttrOfType<FlatSymbolRefAttr>("callee");
}

// The location of the linalg value a value of a graph equals: a value moved
// from a thread is the value the launched subgraph returns there.
Location valueLocation(Value value) {
  Operation *producer = value.getDefiningOp();
  if (nameOf(producer) == "hart.transfer")
    return valueLocation(producer->getOperand(0));
  if (nameOf(producer) == "hart.launch") {
    // Where the launch computes the result: its location, or for several
    // results theirs fused, in order (the callee may be another layer's).
    Location loc = producer->getLoc();
    if (producer->getNumResults() > 1) {
      auto fused = cast<FusedLoc>(loc);
      if (fused.getLocations().size() != producer->getNumResults())
        llvm::report_fatal_error(
            "gvn-slice: a launch has no location per result");
      loc = fused.getLocations()[cast<OpResult>(value).getResultNumber()];
    }
    return loc;
  }
  return producer->getLoc();
}

std::string valueId(ModuleOp module, Value value) {
  return locationId(valueLocation(value));
}

// A part of a linalg value, computed apart from the rest (a thread's rows):
// its location carries its offsets in the value (gvn.offsets), whose type
// is the origin's (gvn.type).
struct Part {
  SmallVector<int64_t> offsets;
  RankedTensorType whole;
};
std::optional<Part> partOf(Location loc) {
  std::optional<Part> part;
  while (true) {
    if (auto fused = dyn_cast<FusedLoc>(loc)) {
      if (auto meta = dyn_cast_or_null<DictionaryAttr>(fused.getMetadata())) {
        if (auto offsets = meta.getAs<DenseI64ArrayAttr>("gvn.offsets"))
          part = Part{SmallVector<int64_t>(offsets.asArrayRef()), {}};
        if (auto type = meta.getAs<TypeAttr>("gvn.type"); type && part)
          part->whole = cast<RankedTensorType>(type.getValue());
      }
      loc = fused.getLocations().front();
    } else if (auto call = dyn_cast<CallSiteLoc>(loc)) {
      loc = call.getCallee();
    } else {
      break;
    }
  }
  if (part && !part->whole)
    llvm::report_fatal_error("gvn-slice: a part has no origin type");
  return part;
}

// A value the compiler made, no value of the source graph: its location is
// a name with no source under it. The slice computes it from its inputs.
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

// An original slice of a graph argument, which the slice keeps: the thread
// may read it in place, at the argument's strides.
bool argumentSlice(Operation *op) {
  return isa<tensor::ExtractSliceOp>(op) &&
         isa<BlockArgument>(op->getOperand(0));
}

// A logical view of the original graph, rather than a partition or padding
// view.
bool originalView(Operation *op) {
  if (!isa<tensor::ExtractSliceOp, tensor::ExpandShapeOp,
           tensor::CollapseShapeOp>(op))
    return false;
  Location loc = op->getLoc();
  while (true) {
    if (auto fused = dyn_cast<FusedLoc>(loc)) {
      if (auto origin = dyn_cast_or_null<DictionaryAttr>(fused.getMetadata())) {
        auto name = origin.getAs<StringAttr>("gvn.operation");
        auto type = origin.getAs<TypeAttr>("gvn.type");
        if (name && type)
          return name.getValue() == op->getName().getStringRef() &&
                 type.getValue() == op->getResult(0).getType();
      }
      loc = fused.getLocations().front();
    } else if (auto call = dyn_cast<CallSiteLoc>(loc)) {
      loc = call.getCallee();
    } else {
      break;
    }
  }
  return false;
}

// Whether `op` holds an operation of the bank layer.
bool holdsBanks(Operation *op) {
  bool banks = false;
  op->walk([&](Operation *nested) {
    StringRef dialect = nested->getName().getDialectNamespace();
    banks |= dialect == "bbacc" || dialect.ends_with("_ball");
  });
  return banks;
}

// Whether the value of `op` is a computation of the graph on the CPU, which
// the GPU computes too: a linalg operation (but a fill) or a frontend one.
bool computed(Operation *op) {
  StringRef dialect = op->getName().getDialectNamespace();
  return (dialect == "linalg" && nameOf(op) != "linalg.fill") ||
         (dialect == "frontend" && !nameOf(op).starts_with("frontend.trace"));
}

class SlicePass : public PassWrapper<SlicePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SlicePass)
  SlicePass() = default;
  SlicePass(const SlicePass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "gvn-slice"; }
  StringRef getDescription() const final {
    return "Keep one slice of a checked hart graph, its inputs read from "
           "gvn_in and its output written to gvn_out";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  Option<std::string> function{*this, "function",
                               llvm::cl::desc("The function to slice")};
  Option<unsigned> index{*this, "index", llvm::cl::desc("The slice to keep")};
  Option<bool> iterate{*this, "iterate",
                       llvm::cl::desc("Keep one iteration of the tile loops"),
                       llvm::cl::init(false)};
  Option<std::string> layout{*this, "layout",
                             llvm::cl::desc("The layout file to write")};
  Option<int64_t> batch{
      *this, "batch",
      llvm::cl::desc("The tile iterations of the batch axis an iteration "
                     "runs (iterate)")};

  ModuleOp module;
  OpBuilder *builder = nullptr;
  LocationAttr loc;
  int64_t slots = 0;
  llvm::json::Array entries;
  // The buffer of the checked value (iterate).
  Value outputBuffer;
  // The largest batch of trips a unit runs.
  int64_t batchUsed = 1;

  // Tile iterations of the output writes: a static loop holding some, its
  // trips times the iterations of the units of its body (a batch of its
  // trips an iteration when they have none), or a segment [first, last] of
  // a block, the iterations of its units (one when it has none).
  struct Unit {
    scf::ForOp loop;
    Operation *first = nullptr, *last = nullptr;
    std::vector<Unit> units;
    int64_t size = 1, tiles = 1;
  };

  static bool staticLoop(Operation *op) {
    auto loop = dyn_cast<scf::ForOp>(op);
    return loop && !loop.getNumResults() &&
           getConstantIntValue(loop.getLowerBound()) &&
           getConstantIntValue(loop.getUpperBound()) &&
           getConstantIntValue(loop.getStep());
  }

  static int64_t tripCount(scf::ForOp loop) {
    int64_t lower = *getConstantIntValue(loop.getLowerBound());
    int64_t upper = *getConstantIntValue(loop.getUpperBound());
    int64_t step = *getConstantIntValue(loop.getStep());
    return (upper - lower + step - 1) / step;
  }

  // The units of the output writes `writes` among the operations [first,
  // last] of a block, in order. A segment is a range no bank defined in it
  // outlives, but a bank held from the first writer to the last: that is
  // the setup of every unit. A unit is a segment holding writes, closed (no
  // value it defines is used after it, it writes no DDR but the output), so
  // that it can be left out of an iteration. The other segments run in
  // every iteration of the range: a segment defining banks a unit reads
  // (the first column of a held A group) recomputes its writes there.
  // None when no static loop holds writes or the range is one segment.
  std::vector<Unit> units(Operation *first, Operation *last,
                          ArrayRef<Operation *> writes) {
    Block &block = *first->getBlock();
    SmallVector<Operation *> ops;
    DenseMap<Operation *, int64_t> at;
    for (Operation *op = first;; op = op->getNextNode()) {
      at[op] = ops.size();
      ops.push_back(op);
      if (op == last)
        break;
    }
    int64_t count = ops.size();
    SmallVector<bool> writer(count, false);
    for (Operation *write : writes)
      if (Operation *top = block.findAncestorOpInBlock(*write);
          top && at.count(top))
        writer[at[top]] = true;
    if (!llvm::any_of(llvm::seq<int64_t>(0, count), [&](int64_t i) {
          return writer[i] && staticLoop(ops[i]);
        }))
      return {};
    int64_t firstWriter = llvm::find(writer, true) - writer.begin();
    int64_t lastWriter = count - 1;
    while (!writer[lastWriter])
      --lastWriter;
    // The position of the operation of the range holding `user`; past the
    // range when none does.
    auto position = [&](Operation *user) -> int64_t {
      Operation *top = block.findAncestorOpInBlock(*user);
      return top && at.count(top) ? at[top] : count;
    };
    SmallVector<int64_t> reach(count);
    for (int64_t i = 0; i < count; ++i) {
      reach[i] = i;
      if (!holdsBanks(ops[i]))
        continue;
      for (Value result : ops[i]->getResults()) {
        if (!result.getType().isInteger(64))
          continue;
        int64_t end = i;
        for (Operation *user : result.getUsers())
          end = std::max(end, position(user));
        if (i <= firstWriter && end >= lastWriter)
          continue;
        reach[i] = std::max(reach[i], std::min(end, count - 1));
      }
    }
    auto closed = [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i <= end; ++i) {
        for (Value result : ops[i]->getResults())
          for (Operation *user : result.getUsers())
            if (position(user) > end)
              return false;
        bool output = true;
        ops[i]->walk([&](Operation *op) {
          output &= !nameOf(op).starts_with("bbacc.bank_mvout") ||
                    rootBuffer(op->getOperand(0)) == outputBuffer;
        });
        if (!output)
          return false;
      }
      return true;
    };
    std::vector<Unit> result;
    for (int64_t begin = 0; begin < count;) {
      int64_t end = begin;
      for (int64_t i = begin; i <= end; ++i)
        end = std::max(end, reach[i]);
      if (begin == 0 && end == count - 1 && !(count == 1 && staticLoop(ops[0])))
        return {};
      if (llvm::any_of(llvm::seq<int64_t>(begin, end + 1),
                       [&](int64_t i) { return writer[i]; }) &&
          closed(begin, end)) {
        Unit unit;
        if (begin == end && staticLoop(ops[begin])) {
          unit.loop = cast<scf::ForOp>(ops[begin]);
          Block *loopBody = unit.loop.getBody();
          if (&loopBody->front() != loopBody->getTerminator())
            unit.units =
                units(&loopBody->front(),
                      loopBody->getTerminator()->getPrevNode(), writes);
          int64_t trips = tripCount(unit.loop);
          if (unit.units.empty()) {
            int64_t trip = std::min<int64_t>(batch, trips);
            batchUsed = std::max(batchUsed, trip);
            unit.size = (trips + trip - 1) / trip;
            unit.tiles = trips;
          } else {
            unit.size = unit.tiles = 0;
            for (const Unit &nested : unit.units) {
              unit.size += trips * nested.size;
              unit.tiles += trips * nested.tiles;
            }
          }
        } else {
          unit.first = ops[begin];
          unit.last = ops[end];
          unit.units = units(unit.first, unit.last, writes);
          if (!unit.units.empty()) {
            unit.size = unit.tiles = 0;
            for (const Unit &nested : unit.units) {
              unit.size += nested.size;
              unit.tiles += nested.tiles;
            }
          }
        }
        result.push_back(std::move(unit));
      }
      begin = end + 1;
    }
    return result;
  }

  Value global(StringRef name, Type element) {
    return memref::GetGlobalOp::create(
        *builder, loc, MemRefType::get({externalSize}, element), name);
  }

  Value tableWord(int64_t slot) {
    Value word = memref::LoadOp::create(
        *builder, loc, global("gvn_table", builder->getI64Type()),
        ValueRange{arith::ConstantIndexOp::create(*builder, loc, slot)});
    return arith::IndexCastOp::create(*builder, loc, builder->getIndexType(),
                                      word);
  }

  // A value of `type` read from gvn_in: its offset and dynamic sizes are the
  // words of a new slot of gvn_table.
  Value readInput(RankedTensorType type, llvm::json::Object entry) {
    int64_t slot = slots;
    slots += 1 + type.getNumDynamicDims();
    Value offset = tableWord(slot);
    SmallVector<Value> sizes;
    for (int64_t dynamic = 0; dynamic < type.getNumDynamicDims(); ++dynamic)
      sizes.push_back(tableWord(slot + 1 + dynamic));
    Value view = memref::ViewOp::create(
        *builder, loc, MemRefType::get(type.getShape(), type.getElementType()),
        global("gvn_in", builder->getI8Type()), offset, sizes);
    entry["slot"] = slot;
    entry["type"] = typeName(type);
    entry["element"] = elementName(type.getElementType());
    entries.push_back(std::move(entry));
    return bufferization::ToTensorOp::create(*builder, loc, type, view, true);
  }

  // An uninitialized tensor of `type`, its dynamic sizes from a new slot of
  // gvn_table.
  Value uninitialized(RankedTensorType type, llvm::json::Object entry) {
    int64_t slot = slots;
    slots += type.getNumDynamicDims();
    SmallVector<Value> sizes;
    for (int64_t dynamic = 0; dynamic < type.getNumDynamicDims(); ++dynamic)
      sizes.push_back(tableWord(slot + dynamic));
    entry["slot"] = slot;
    entry["type"] = typeName(type);
    entries.push_back(std::move(entry));
    return tensor::EmptyOp::create(*builder, loc, type.getShape(),
                                   type.getElementType(), sizes);
  }

  // Removes every top-level operation of `body` but `kept`, in reverse.
  void removeAllBut(Block &body, const llvm::SetVector<Operation *> &kept) {
    SmallVector<Operation *> removed;
    for (Operation &op : body.without_terminator())
      if (!kept.contains(&op))
        removed.push_back(&op);
    for (Operation *op : llvm::reverse(removed)) {
      if (!op->use_empty()) {
        op->emitOpError("is removed from the slice but still used by ")
            << **op->getUsers().begin();
        llvm::report_fatal_error("gvn-slice: malformed slice");
      }
      op->erase();
    }
  }

  void runOnOperation() override {
    module = getOperation();
    if (iterate && batch < 1)
      llvm::report_fatal_error("gvn-slice: iterate needs batch >= 1");
    MLIRContext *context = module.getContext();
    OpBuilder moduleBuilder(context);
    builder = &moduleBuilder;
    loc = module.getLoc();

    auto sliced = module.lookupSymbol<func::FuncOp>(function);
    if (!sliced)
      llvm::report_fatal_error("gvn-slice: no such function");
    // Fold constant tile-bound arithmetic without rewriting bank operations
    // or the check boundaries used to compare a slice with the whole graph.
    OperationFolder folder(context);
    SmallVector<Operation *> arithmetic;
    sliced.walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "arith" &&
          !isa<arith::ConstantOp>(op))
        arithmetic.push_back(op);
    });
    for (Operation *op : arithmetic)
      (void)folder.tryToFold(op);
    SmallVector<::gvn::CheckOp> checks;
    sliced.walk([&](::gvn::CheckOp check) { checks.push_back(check); });
    if (index >= checks.size())
      llvm::report_fatal_error("gvn-slice: no such slice");
    Block &body = sliced.front();
    DenseMap<Operation *, int64_t> position;
    for (Operation &op : body)
      position[&op] = position.size();
    int64_t begin = index ? position[checks[index - 1]] : -1;
    int64_t end = position[checks[index]];

    // The values of the earlier checks and of the cycle markers observing
    // them, by check.
    DenseMap<Value, unsigned> inputs;
    for (unsigned check = 0; check < index; ++check) {
      SmallVector<Value> values{checks[check].getValue()};
      while (!values.empty()) {
        Value value = values.pop_back_val();
        inputs[value] = check;
        for (Operation *user : value.getUsers())
          if (nameOf(user) == "frontend.trace_end")
            values.push_back(user->getResult(0));
      }
    }

    // The top-level operations touching each buffer, and those of them that
    // only read it into banks.
    DenseMap<Value, SmallVector<Operation *>> touching;
    DenseMap<Value, DenseSet<Operation *>> writing;
    for (Operation &op : body.without_terminator())
      op.walk([&](Operation *nested) {
        for (Value operand : nested->getOperands())
          if (isa<MemRefType>(operand.getType()) &&
              !nameOf(nested).starts_with("bbacc.bank_mvin"))
            writing[rootBuffer(operand)].insert(&op);
        for (Value operand : nested->getOperands()) {
          if (!isa<MemRefType>(operand.getType()))
            continue;
          auto &users = touching[rootBuffer(operand)];
          if (users.empty() || users.back() != &op)
            users.push_back(&op);
        }
      });

    llvm::SetVector<Operation *> kept;
    SmallVector<Operation *> worklist;
    auto keep = [&](Operation *op) {
      if (kept.insert(op))
        worklist.push_back(op);
    };
    // With iterate, the slice is a batch of tile iterations: the units of
    // the output writes in the span (units). Invariant bank setup stays
    // around each tile. The CPU computations the iteration reads are read
    // from gvn_in instead of computed.
    std::vector<Unit> tiles;
    SmallVector<Operation *> outputWrites;
    if (iterate) {
      // A tile must contain the output writes, not just a DMA-loading loop
      // preceding an unrolled computation.
      auto tensor =
          checks[index].getValue().getDefiningOp<bufferization::ToTensorOp>();
      if (tensor) {
        outputBuffer = rootBuffer(tensor.getBuffer());
        sliced.walk([&](Operation *op) {
          if (nameOf(op).starts_with("bbacc.bank_mvout") &&
              rootBuffer(op->getOperand(0)) == outputBuffer)
            outputWrites.push_back(op);
        });
      }
      tiles = units(index ? checks[index - 1]->getNextNode() : &body.front(),
                    checks[index], outputWrites);
    }
    if (!tiles.empty()) {
      for (Operation &op : body.without_terminator())
        if (position[&op] > begin && position[&op] <= end &&
            llvm::any_of(outputWrites, [&](Operation *write) {
              return op.isAncestor(write);
            }))
          keep(&op);
      // A segment is guarded whole.
      for (const Unit &unit : tiles)
        for (Operation *op = unit.first; op; op = op->getNextNode()) {
          keep(op);
          if (op == unit.last)
            break;
        }
    } else
      for (Operation &op : body.without_terminator())
        if (position[&op] > begin && position[&op] <= end &&
            !isa<::gvn::CheckOp>(op))
          keep(&op);
    llvm::SetVector<Value> cpuValues;
    std::set<unsigned> read;
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (nameOf(op) == "frontend.trace_end")
        for (Operation *start = op->getPrevNode(); start;
             start = start->getPrevNode())
          if (nameOf(start) == "frontend.trace_start" &&
              start->getAttr("id") == op->getAttr("id") &&
              start->getAttr("id_path") == op->getAttr("id_path")) {
            keep(start);
            break;
          }
      op->walk([&](Operation *nested) {
        for (Value operand : nested->getOperands()) {
          auto input = inputs.find(operand);
          if (input != inputs.end()) {
            read.insert(input->second);
            continue;
          }
          Operation *producer = operand.getDefiningOp();
          if (!producer)
            continue;
          Operation *top = body.findAncestorOpInBlock(*producer);
          if (top == op)
            continue;
          if (!tiles.empty() && top == producer &&
              isa<RankedTensorType>(operand.getType()) && computed(producer)) {
            cpuValues.insert(operand);
            continue;
          }
          if (position[top] > end) {
            op->emitOpError("reads a value defined after its slice");
            llvm::report_fatal_error("gvn-slice: malformed slice");
          }
          keep(top);
          // Before bank SSA lowering, users may update an allocation in
          // place without threading the returned bank value to later reads.
          if (nameOf(producer) == "bbacc.bank_alloc")
            for (Operation *user : operand.getUsers()) {
              Operation *writer = body.findAncestorOpInBlock(*user);
              if (position[writer] <= end)
                keep(writer);
            }
          // Before the slice, another reader of the buffer into its own
          // banks (an earlier check's) is not needed.
          if (isa<MemRefType>(operand.getType()))
            for (Operation *user : touching[rootBuffer(operand)])
              if (position[user] <= end &&
                  (position[user] > begin ||
                   writing[rootBuffer(operand)].contains(user)))
                keep(user);
        }
      });
    }
    // The banks the slice allocates are released as they are in f.
    for (Operation *op : SmallVector<Operation *>(kept.getArrayRef()))
      for (Value result : op->getResults())
        for (Operation *user : result.getUsers())
          if (nameOf(user) == "bbacc.bank_release" && user->getBlock() == &body)
            kept.insert(user);
    // A cycle trace the slice starts but does not end has no start: its end
    // reads a value of a later slice.
    for (Operation *op : SmallVector<Operation *>(kept.getArrayRef())) {
      if (nameOf(op) != "frontend.trace_start")
        continue;
      bool ended = llvm::any_of(kept, [&](Operation *end) {
        return nameOf(end) == "frontend.trace_end" &&
               end->getAttr("id") == op->getAttr("id") &&
               end->getAttr("id_path") == op->getAttr("id_path");
      });
      if (!ended)
        kept.remove(op);
    }

    builder->setInsertionPointToStart(module.getBody());
    for (auto [name, element] :
         {std::pair<StringRef, Type>{"gvn_in", builder->getI8Type()},
          {"gvn_out", builder->getI8Type()},
          {"gvn_table", builder->getI64Type()}})
      memref::GlobalOp::create(
          *builder, loc, name, builder->getStringAttr("public"),
          MemRefType::get({externalSize}, element), Attribute(), false,
          builder->getI64IntegerAttr(64));

    // The checks of f the slice reads.
    builder->setInsertionPointToStart(&body);
    for (unsigned check : read) {
      auto type = cast<RankedTensorType>(checks[check].getValue().getType());
      Value value = readInput(
          type, llvm::json::Object{{"kind", "value"},
                                   {"id", checks[check].getId().str()}});
      for (auto [input, inputCheck] : inputs)
        if (inputCheck == check)
          input.replaceUsesWithIf(value, [&](OpOperand &use) {
            return kept.contains(body.findAncestorOpInBlock(*use.getOwner()));
          });
    }

    // The CPU computations an iteration reads.
    for (Value value : cpuValues) {
      Value input =
          readInput(cast<RankedTensorType>(value.getType()),
                    llvm::json::Object{
                        {"kind", "value"},
                        {"id", locationId(value.getDefiningOp()->getLoc())}});
      value.replaceUsesWithIf(input, [&](OpOperand &use) {
        return kept.contains(body.findAncestorOpInBlock(*use.getOwner()));
      });
    }

    // One iteration of the tile loops: its index in the slot of gvn_table,
    // in row-major order of the loops, which have static trip counts.
    int64_t iterations = 1, iterationSlot = -1, tileIterations = 1,
            iterationBatch = 1;
    auto selectIterations = [&](scf::ForOp loop, Value lower, Value upper) {
      if (!loop->hasAttr("hart.stream")) {
        loop.setLowerBound(lower);
        loop.setUpperBound(upper);
        return;
      }
      Value step = loop.getStep();
      Value count = arith::CeilDivSIOp::create(
          *builder, loc, arith::SubIOp::create(*builder, loc, upper, lower),
          step);
      loop.setLowerBound(arith::ConstantIndexOp::create(*builder, loc, 0));
      loop.setUpperBound(count);
      loop.setStep(arith::ConstantIndexOp::create(*builder, loc, 1));
      OpBuilder::InsertionGuard guard(*builder);
      builder->setInsertionPointToStart(loop.getBody());
      auto scaled =
          arith::MulIOp::create(*builder, loc, loop.getInductionVar(), step);
      Value original = arith::AddIOp::create(*builder, loc, lower, scaled);
      loop.getInductionVar().replaceUsesWithIf(
          original, [&](OpOperand &use) { return use.getOwner() != scaled; });
    };
    SmallVector<scf::ForOp> columnTiles;
    int64_t columnCount = 0;
    if (iterate && !outputWrites.empty() &&
        !cast<RankedTensorType>(checks[index].getValue().getType())
             .hasStaticShape()) {
      // Dynamic row tails have separate loops, but share the same static
      // column tiling. Select a column in every row tail instead of keeping
      // the complete matrix in one stimulus.
      for (Operation *write : outputWrites) {
        scf::ForOp selected;
        for (Operation *parent = write->getParentOp(); parent != sliced;
             parent = parent->getParentOp()) {
          auto loop = dyn_cast<scf::ForOp>(parent);
          if (!loop)
            continue;
          auto lower = getConstantIntValue(loop.getLowerBound());
          auto upper = getConstantIntValue(loop.getUpperBound());
          auto step = getConstantIntValue(loop.getStep());
          auto rows = loop->getParentOfType<scf::ForOp>();
          if (!lower || !upper || !step || !rows ||
              getConstantIntValue(rows.getUpperBound()))
            continue;
          int64_t count = (*upper - *lower + *step - 1) / *step;
          if (count <= 1)
            continue;
          if (columnCount && columnCount != count)
            break;
          columnCount = count;
          selected = loop;
          break;
        }
        if (!selected) {
          columnTiles.clear();
          break;
        }
        if (!llvm::is_contained(columnTiles, selected))
          columnTiles.push_back(selected);
      }
    }
    if (!columnTiles.empty()) {
      tileIterations = columnCount;
      iterationBatch = std::min<int64_t>(batch, columnCount);
      iterations = (columnCount + iterationBatch - 1) / iterationBatch;
      iterationSlot = slots++;
      builder->setInsertionPointToStart(&body);
      Value iteration = tableWord(iterationSlot);
      for (scf::ForOp loop : columnTiles) {
        builder->setInsertionPoint(loop);
        Value span = arith::MulIOp::create(
            *builder, loc, loop.getStep(),
            arith::ConstantIndexOp::create(*builder, loc, iterationBatch));
        Value lower = arith::AddIOp::create(
            *builder, loc, loop.getLowerBound(),
            arith::MulIOp::create(*builder, loc, iteration, span));
        Value upper = arith::MinSIOp::create(
            *builder, loc, arith::AddIOp::create(*builder, loc, lower, span),
            loop.getUpperBound());
        selectIterations(loop, lower, upper);
      }
    } else if (!tiles.empty()) {
      tileIterations = 0;
      for (const Unit &unit : tiles)
        tileIterations += unit.tiles;
      iterationBatch = batch < tileIterations ? batchUsed : tileIterations;
    }
    // A batch of every tile iteration runs the span whole, in one iteration.
    if (columnTiles.empty() && !tiles.empty() && batch < tileIterations) {
      iterations = 0;
      for (const Unit &unit : tiles)
        iterations += unit.size;
      iterationSlot = slots++;
      builder->setInsertionPointToStart(&body);
      Value iteration = tableWord(iterationSlot);
      auto constant = [&](int64_t value) -> Value {
        return arith::ConstantIndexOp::create(*builder, loc, value);
      };
      // The units at `local` of a level `active` selects run: a loop its
      // trip (a batch of trips without units), none when not selected; a
      // segment beside others runs in an scf.if.
      std::function<void(ArrayRef<Unit>, Value, Value)> select =
          [&](ArrayRef<Unit> units, Value local, Value active) {
            int64_t first = 0;
            for (const Unit &unit : units) {
              scf::ForOp loop = unit.loop;
              builder->setInsertionPoint(loop ? loop.getOperation()
                                              : unit.first);
              Value at =
                  arith::SubIOp::create(*builder, loc, local, constant(first));
              Value selected = active;
              if (units.size() > 1)
                selected = arith::AndIOp::create(
                    *builder, loc, active,
                    arith::AndIOp::create(
                        *builder, loc,
                        arith::CmpIOp::create(*builder, loc,
                                              arith::CmpIPredicate::uge, local,
                                              constant(first)),
                        arith::CmpIOp::create(*builder, loc,
                                              arith::CmpIPredicate::ult, local,
                                              constant(first + unit.size))));
              int64_t inner = 0;
              for (const Unit &nested : unit.units)
                inner += nested.size;
              if (loop) {
                Value lowerBound = loop.getLowerBound();
                int64_t step = *getConstantIntValue(loop.getStep());
                Value trip = at, rest;
                if (unit.units.empty()) {
                  step *= std::min<int64_t>(batch, tripCount(loop));
                } else {
                  trip = arith::DivUIOp::create(*builder, loc, at,
                                                constant(inner));
                  rest = arith::RemUIOp::create(*builder, loc, at,
                                                constant(inner));
                }
                Value lower = arith::AddIOp::create(
                    *builder, loc, lowerBound,
                    arith::MulIOp::create(*builder, loc, trip, constant(step)));
                Value upper = arith::MinSIOp::create(
                    *builder, loc,
                    arith::AddIOp::create(*builder, loc, lower, constant(step)),
                    loop.getUpperBound());
                lower = arith::SelectOp::create(*builder, loc, selected, lower,
                                                lowerBound);
                upper = arith::SelectOp::create(*builder, loc, selected, upper,
                                                lowerBound);
                selectIterations(loop, lower, upper);
                if (!unit.units.empty())
                  select(unit.units, rest, selected);
              } else {
                if (!unit.units.empty())
                  select(unit.units, at, selected);
                if (units.size() > 1) {
                  SmallVector<Operation *> guarded;
                  for (Operation *op = unit.first;; op = op->getNextNode()) {
                    guarded.push_back(op);
                    if (op == unit.last)
                      break;
                  }
                  builder->setInsertionPoint(unit.first);
                  auto branch =
                      scf::IfOp::create(*builder, loc, selected, false);
                  for (Operation *op : guarded)
                    op->moveBefore(
                        branch.getThenRegion().front().getTerminator());
                }
              }
              first += unit.size;
            }
          };
      builder->setInsertionPointToStart(&body);
      select(tiles, iteration,
             arith::ConstantIntOp::create(*builder, loc, 1, 1));
    }

    // The value of check k reaches gvn_out. A subgraph launched on a
    // hardware thread returns it: its graph moves it from the thread into
    // gvn_out (sliceGraph). Otherwise its buffer, allocated or the
    // destination it is given, becomes a view of gvn_out.
    Value output = checks[index].getValue();
    auto outputType = cast<RankedTensorType>(output.getType());
    SmallVector<Operation *> launches;
    module.walk([&](Operation *op) {
      if (nameOf(op) == "hart.launch" && calleeOf(op).getValue() == function)
        launches.push_back(op);
    });
    // A subgraph several layers launch (one function, convert-linalg-to-hart)
    // is sliced at its first launch: the others run the same code on other
    // data, and its values carry the locations of the first.
    int64_t returned = -1;
    for (OpOperand &result : body.getTerminator()->getOpOperands())
      if (result.get() == output)
        returned = result.getOperandNumber();
    if (!launches.empty() && returned < 0) {
      if (!checks[index].getId().starts_with("memory:"))
        llvm::report_fatal_error("gvn-slice: a launched subgraph does not "
                                 "return its checked value");
      // The isolated producer returns its DDR checkpoint in place of its
      // unused first result. Keep the normal launch/transfer output contract.
      returned = 0;
      auto functionType = sliced.getFunctionType();
      unsigned destination =
          sliced.getNumArguments() - functionType.getNumResults();
      // The original destination is empty and may still have logical-shape
      // views. Preserve those views while returning the physical checkpoint.
      Operation *launch = launches.front();
      if (!launch->getOperand(destination).getDefiningOp<tensor::EmptyOp>())
        llvm::report_fatal_error(
            "gvn-slice: checkpoint destination is not empty");
      builder->setInsertionPointToStart(&body);
      auto logicalType =
          cast<RankedTensorType>(sliced.getArgument(destination).getType());
      Value logical = tensor::EmptyOp::create(
          *builder, loc, logicalType.getShape(), logicalType.getElementType());
      sliced.getArgument(destination).replaceAllUsesWith(logical);
      sliced.getArgument(destination).setType(outputType);
      sliced.removeArgAttr(destination, "bufferization.buffer_layout");
      SmallVector<Type> inputs(functionType.getInputs());
      SmallVector<Type> results(functionType.getResults());
      inputs[destination] = outputType;
      results[0] = outputType;
      sliced.setFunctionType(builder->getFunctionType(inputs, results));
      auto allocation = output.getDefiningOp<bufferization::ToTensorOp>()
                            .getBuffer()
                            .getDefiningOp<memref::AllocOp>();
      builder->setInsertionPoint(allocation);
      Value target = bufferization::ToBufferOp::create(
          *builder, loc, allocation.getType(), sliced.getArgument(destination));
      allocation.getResult().replaceAllUsesWith(target);
      allocation.erase();
      builder->setInsertionPoint(body.getTerminator());
      auto materialize = bufferization::MaterializeInDestinationOp::create(
          *builder, loc, output, sliced.getArgument(destination));
      kept.insert(materialize);
      body.getTerminator()->setOperand(0, materialize.getResult());
      builder->setInsertionPoint(launch);
      Value checkpoint = tensor::EmptyOp::create(
          *builder, loc, outputType.getShape(), outputType.getElementType());
      launch->setOperand(destination, checkpoint);
      launch->getResult(0).setType(outputType);
      for (Operation *user : launch->getResult(0).getUsers())
        if (nameOf(user) == "hart.transfer") {
          builder->setInsertionPoint(user);
          user->setOperand(
              1, tensor::EmptyOp::create(*builder, loc, outputType.getShape(),
                                         outputType.getElementType()));
          user->getResult(0).setType(outputType);
        }
    }
    if (launches.empty()) {
      Operation *buffer = output.getDefiningOp()->getOperand(0).getDefiningOp();
      auto bufferType = cast<MemRefType>(buffer->getResult(0).getType());
      builder->setInsertionPoint(buffer);
      SmallVector<Value> dynamicSizes;
      if (auto allocation = dyn_cast<memref::AllocOp>(buffer)) {
        dynamicSizes.assign(allocation.getDynamicSizes().begin(),
                            allocation.getDynamicSizes().end());
      } else if (auto toBuffer = dyn_cast<bufferization::ToBufferOp>(buffer)) {
        auto empty = toBuffer.getTensor().getDefiningOp<tensor::EmptyOp>();
        if (!empty)
          llvm::report_fatal_error(
              "gvn-slice: a checked tensor has no empty allocation");
        dynamicSizes.assign(empty.getDynamicSizes().begin(),
                            empty.getDynamicSizes().end());
      } else {
        llvm::report_fatal_error("gvn-slice: a checked value is no allocation");
      }
      Value view = memref::ViewOp::create(
          *builder, loc, bufferType, global("gvn_out", builder->getI8Type()),
          arith::ConstantIndexOp::create(*builder, loc, 0), dynamicSizes);
      buffer->getResult(0).replaceAllUsesWith(view);
      buffer->erase();
    }

    // Results f does not compute: a launched subgraph returns the
    // destination it is given unwritten (its results are its destinations);
    // Standalone slices report through gvn_out and return an unused empty
    // tensor, so ownership lowering does not clone the external output.
    Operation *terminator = body.getTerminator();
    builder->setInsertionPoint(terminator);
    for (OpOperand &result : terminator->getOpOperands()) {
      Operation *producer = result.get().getDefiningOp();
      if (!launches.empty() && producer &&
          kept.contains(body.findAncestorOpInBlock(*producer)))
        continue;
      if (!launches.empty()) {
        result.set(sliced.getArgument(sliced.getNumArguments() -
                                      terminator->getNumOperands() +
                                      result.getOperandNumber()));
        continue;
      }
      auto type = cast<RankedTensorType>(result.get().getType());
      if (!type.hasStaticShape())
        llvm::report_fatal_error("gvn-slice: an uncomputed result of dynamic "
                                 "shape");
      result.set(tensor::EmptyOp::create(*builder, loc, type.getShape(),
                                         type.getElementType()));
    }
    // Every operation inserted at the top is kept.
    for (Operation &op : body.without_terminator())
      if (!position.count(&op))
        kept.insert(&op);

    llvm::json::Object description{
        {"function", std::string(function)},
        {"index", static_cast<int64_t>(index)},
        {"slices", static_cast<int64_t>(checks.size())},
        {"iterations", iterations},
        {"tile_iterations", tileIterations},
        {"iteration_batch", iterationBatch},
        {"iteration_slot", iterationSlot},
        {"output", llvm::json::Object{
                       {"id", checks[index].getId().str()},
                       {"type", typeName(outputType)},
                       {"element", elementName(outputType.getElementType())}}}};
    removeAllBut(body, kept);

    // The graph launching f on a hardware thread keeps that launch.
    if (!launches.empty())
      sliceGraph(launches.front(), returned, description);

    // The marks of the other functions.
    module.walk([](::gvn::CheckOp check) { check.erase(); });

    description["entries"] = std::move(entries);
    description["table_words"] = slots;
    std::error_code error;
    llvm::raw_fd_ostream file(layout, error);
    if (error)
      llvm::report_fatal_error("gvn-slice: cannot write the layout file");
    file << llvm::formatv("{0:2}", llvm::json::Value(std::move(description)))
         << "\n";
  }

  void sliceGraph(Operation *launch, int64_t returned,
                  llvm::json::Object &description) {
    description["thread"] =
        launch->getAttrOfType<IntegerAttr>("thread").getInt();
    auto graph = launch->getParentOfType<func::FuncOp>();
    Block &body = graph.front();
    DenseSet<Operation *> original;
    for (Operation &op : body)
      original.insert(&op);
    // The transfer of the checked value from the thread writes gvn_out.
    Operation *back = nullptr;
    for (Operation *user : launch->getResult(returned).getUsers())
      if (nameOf(user) == "hart.transfer")
        back = user;
    if (!back)
      llvm::report_fatal_error("gvn-slice: a checked value never leaves its "
                               "thread");
    auto type = cast<RankedTensorType>(back->getResult(0).getType());
    builder->setInsertionPoint(back);
    // A thread computing a partition of the checked value (rows or columns
    // split over threads by convert-linalg-to-hart) returns only its part;
    // the graph inserts it into the full value. gvn_out holds the full value,
    // as the GPU records it, and the transfer writes the partition at the
    // coordinates of that insertion. The full sizes are words of gvn_table,
    // and the view of gvn_out is formed before the launch: lower-hart moves
    // the destination of a partition before the launch writing it, and can
    // move only pure operations (the coordinates, the subview), not loads.
    // The insertion of a partition is located at the checked value (the
    // split result takes its loc); an insertion located elsewhere builds
    // another value from the whole checked value (a stack of two).
    std::string id = description.getObject("output")->getString("id")->str();
    tensor::InsertSliceOp insertion;
    for (Operation *user : back->getResult(0).getUsers())
      if (auto slice = dyn_cast<tensor::InsertSliceOp>(user);
          slice && slice.getSource() == back->getResult(0) &&
          locationId(slice->getLoc()) == id)
        insertion = slice;
    Value view;
    if (insertion) {
      builder->setInsertionPoint(launch);
      RankedTensorType full = insertion.getResultType();
      int64_t slot = slots;
      slots += full.getNumDynamicDims();
      SmallVector<Value> sizes;
      for (int64_t dynamic = 0; dynamic < full.getNumDynamicDims(); ++dynamic)
        sizes.push_back(tableWord(slot + dynamic));
      llvm::json::Object output{
          {"kind", "output"}, {"slot", slot}, {"type", typeName(full)}};
      // The full value can be a box of a larger one (some heads of a cache
      // the graph joins), which the GPU records: its offsets there.
      Value destination = insertion.getDest();
      while (auto inner = destination.getDefiningOp<tensor::InsertSliceOp>())
        destination = inner.getDest();
      if (auto box = destination.getDefiningOp<tensor::ExtractSliceOp>();
          box && box.hasUnitStride() && box.getSourceType().hasStaticShape() &&
          box.getSourceType().getRank() == full.getRank() &&
          !ShapedType::isDynamicShape(box.getStaticOffsets())) {
        output["offsets"] = llvm::json::Array(box.getStaticOffsets());
        output["within"] = typeName(box.getSourceType());
      }
      entries.push_back(std::move(output));
      for (Value coordinate : insertion->getOperands().drop_front(2))
        if (Operation *producer = coordinate.getDefiningOp();
            producer && !producer->isBeforeInBlock(back)) {
          insertion.emitOpError("places the checked partition with a value "
                                "computed after its transfer");
          llvm::report_fatal_error("gvn-slice: malformed slice");
        }
      Value whole = memref::ViewOp::create(
          *builder, loc,
          MemRefType::get(full.getShape(), full.getElementType()),
          global("gvn_out", builder->getI8Type()),
          arith::ConstantIndexOp::create(*builder, loc, 0), sizes);
      builder->setInsertionPoint(back);
      auto partType = memref::SubViewOp::inferRankReducedResultType(
          type.getShape(), cast<MemRefType>(whole.getType()),
          insertion.getMixedOffsets(), insertion.getMixedSizes(),
          insertion.getMixedStrides());
      view = memref::SubViewOp::create(
          *builder, loc, partType, whole, insertion.getMixedOffsets(),
          insertion.getMixedSizes(), insertion.getMixedStrides());
    } else if (std::optional<Part> part =
                   partOf(valueLocation(launch->getResult(returned)))) {
      // A part of the checked value whose consumers read it directly: gvn_out
      // holds the whole value, as the GPU records it, the transfer writes the
      // part at its offsets.
      builder->setInsertionPoint(launch);
      entries.push_back(llvm::json::Object{{"kind", "output"},
                                           {"slot", slots},
                                           {"type", typeName(part->whole)}});
      Value whole = memref::ViewOp::create(
          *builder, loc,
          MemRefType::get(part->whole.getShape(), part->whole.getElementType()),
          global("gvn_out", builder->getI8Type()),
          arith::ConstantIndexOp::create(*builder, loc, 0), ValueRange{});
      builder->setInsertionPoint(back);
      SmallVector<int64_t> ones(type.getRank(), 1);
      auto partType = memref::SubViewOp::inferResultType(
          cast<MemRefType>(whole.getType()), part->offsets, type.getShape(),
          ones);
      view = memref::SubViewOp::create(*builder, loc,
                                       cast<MemRefType>(partType), whole,
                                       part->offsets, type.getShape(), ones);
    } else {
      // The thread returns the whole value.
      SmallVector<Value> sizes;
      for (int64_t dim = 0; dim < type.getRank(); ++dim)
        if (type.isDynamicDim(dim))
          sizes.push_back(
              tensor::DimOp::create(*builder, loc, back->getOperand(1), dim));
      view = memref::ViewOp::create(
          *builder, loc,
          MemRefType::get(type.getShape(), type.getElementType()),
          global("gvn_out", builder->getI8Type()),
          arith::ConstantIndexOp::create(*builder, loc, 0), sizes);
    }
    back->setOperand(1, bufferization::ToTensorOp::create(*builder, loc, type,
                                                          view, true, true));
    builder->setInsertionPointAfter(back);
    auto store = bufferization::MaterializeInDestinationOp::create(
        *builder, loc, back->getResult(0), view);
    store.setRestrict(true);
    store.setWritable(true);

    llvm::SetVector<Operation *> kept;
    SmallVector<Operation *> worklist;
    // The sources of the transfers that the graph computes (on its CPU or on
    // another thread) are read from gvn_in.
    llvm::SetVector<OpOperand *> sources;
    auto keep = [&](Operation *op) {
      Operation *top = body.findAncestorOpInBlock(*op);
      if (kept.insert(top))
        worklist.push_back(top);
    };
    keep(store);
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      op->walk([&](Operation *nested) {
        for (OpOperand &operand : nested->getOpOperands()) {
          if (sources.contains(&operand))
            continue;
          if (nameOf(nested) == "hart.transfer" &&
              operand.getOperandNumber() == 0 &&
              operand.get().getDefiningOp() &&
              operand.get().getDefiningOp() != launch) {
            // Keep the original views: their locations can name the full
            // producer even when this thread consumes only a row partition.
            OpOperand *source = &operand;
            while (Operation *view = source->get().getDefiningOp()) {
              if (!isa<tensor::ExtractSliceOp, tensor::ExpandShapeOp,
                       tensor::CollapseShapeOp>(view) ||
                  (originalView(view) && !argumentSlice(view)))
                break;
              keep(view);
              source = &view->getOpOperand(0);
            }
            // A compiler-generated DMA producer has no mathematical value
            // in the source graph. Keep that producer and bind its inputs.
            if (Operation *producer = source->get().getDefiningOp();
                producer && nameOf(producer) == "hart.launch") {
              auto callee = module.lookupSymbol<func::FuncOp>(
                  calleeOf(producer).getValue());
              bool computes = false;
              callee.walk([&](Operation *operation) {
                computes |=
                    operation->getName().getDialectNamespace().ends_with(
                        "_ball") ||
                    nameOf(operation) == "bbacc.bank_run_operator";
              });
              if (!computes) {
                keep(producer);
                continue;
              }
            }
            // A constant (a zero bias) or a value the compiler made is part
            // of the slice.
            if (Operation *producer = source->get().getDefiningOp();
                producer && (producer->hasTrait<OpTrait::ConstantLike>() ||
                             generated(producer->getLoc()))) {
              keep(producer);
              continue;
            }
            if (source->get().getDefiningOp())
              sources.insert(source);
            continue;
          }
          Operation *producer = operand.get().getDefiningOp();
          if (producer && isa<RankedTensorType>(operand.get().getType()) &&
              !generated(producer->getLoc()) &&
              (computed(producer) ||
               (originalView(producer) && !argumentSlice(producer)))) {
            sources.insert(&operand);
            continue;
          }
          if (producer && !op->isAncestor(producer))
            keep(producer);
        }
      });
    }
    builder->setInsertionPointToStart(&body);
    DenseMap<Value, Value> fixtures;
    for (OpOperand *source : sources) {
      Value originalSource = source->get();
      if (auto found = fixtures.find(originalSource); found != fixtures.end()) {
        source->set(found->second);
        continue;
      }
      auto type = cast<RankedTensorType>(originalSource.getType());
      // Canonicalization may forward a partition directly from its producer.
      // Its insertion into the full result retains the exact crop coordinates.
      // A dynamic size qualifies when it is the size the full result was
      // created with at that dimension (the partition spans it): the crop
      // then takes that dimension of the full value read back.
      auto spansCreated = [](tensor::InsertSliceOp slice) {
        Value destination = slice.getDest();
        while (auto inner = destination.getDefiningOp<tensor::InsertSliceOp>())
          destination = inner.getDest();
        auto created = destination.getDefiningOp<tensor::EmptyOp>();
        if (!created)
          return false;
        for (auto [dimension, size] : llvm::enumerate(slice.getMixedSizes()))
          if (auto value = dyn_cast<Value>(size);
              value && (!created.getType().isDynamicDim(dimension) ||
                        created.getDynamicSize(dimension) != value))
            return false;
        return true;
      };
      tensor::InsertSliceOp insertion;
      for (Operation *user : source->get().getUsers())
        if (auto slice = dyn_cast<tensor::InsertSliceOp>(user);
            slice && slice.getSource() == source->get() &&
            slice.getOffsets().empty() && slice.getStrides().empty() &&
            (slice.getSizes().empty() || spansCreated(slice)))
          insertion = slice;
      if (insertion) {
        Value whole = readInput(
            cast<RankedTensorType>(insertion.getResult().getType()),
            llvm::json::Object{{"kind", "value"},
                               {"graph", graph.getSymName().str()},
                               {"id", locationId(insertion->getLoc())}});
        SmallVector<OpFoldResult> sizes;
        for (auto [dimension, size] :
             llvm::enumerate(insertion.getMixedSizes()))
          sizes.push_back(
              isa<Value>(size)
                  ? OpFoldResult(tensor::DimOp::create(*builder, loc, whole,
                                                       int64_t(dimension))
                                     .getResult())
                  : size);
        source->set(tensor::ExtractSliceOp::create(
            *builder, loc, type, whole, insertion.getMixedOffsets(), sizes,
            insertion.getMixedStrides()));
        fixtures[originalSource] = source->get();
        continue;
      }
      // A part of a value (a thread's heads), its consumers reading it
      // directly: the whole value read back, the part taken of it.
      if (std::optional<Part> part = partOf(valueLocation(source->get()))) {
        Value whole = readInput(
            part->whole,
            llvm::json::Object{{"kind", "value"},
                               {"graph", graph.getSymName().str()},
                               {"id", valueId(module, source->get())}});
        SmallVector<int64_t> ones(type.getRank(), 1);
        source->set(tensor::ExtractSliceOp::create(
            *builder, loc, type, whole,
            getAsIndexOpFoldResult(builder->getContext(), part->offsets),
            getAsIndexOpFoldResult(builder->getContext(), type.getShape()),
            getAsIndexOpFoldResult(builder->getContext(), ones)));
        fixtures[originalSource] = source->get();
        continue;
      }
      Value reshaped = source->get();
      std::string sourceId = valueId(module, source->get());
      // Grouped reductions restore their logical shape through views with
      // the same source location. Views for later computations do not qualify.
      while (true) {
        Operation *view = nullptr;
        for (Operation *user : reshaped.getUsers())
          if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(user) &&
              locationId(user->getLoc()) == sourceId) {
            view = user;
            break;
          }
        if (!view)
          break;
        reshaped = view->getResult(0);
      }
      if (reshaped != source->get() &&
          cast<RankedTensorType>(reshaped.getType()).hasStaticShape()) {
        Value input = readInput(
            cast<RankedTensorType>(reshaped.getType()),
            llvm::json::Object{
                {"kind", "value"},
                {"graph", graph.getSymName().str()},
                {"id", locationId(reshaped.getDefiningOp()->getLoc())}});
        Value shape = arith::ConstantOp::create(
            *builder, loc,
            DenseIntElementsAttr::get(
                RankedTensorType::get({type.getRank()}, builder->getI64Type()),
                type.getShape()));
        source->set(
            tensor::ReshapeOp::create(*builder, loc, type, input, shape));
        fixtures[originalSource] = source->get();
        continue;
      }
      source->set(readInput(
          type, llvm::json::Object{{"kind", "value"},
                                   {"graph", graph.getSymName().str()},
                                   {"id", valueId(module, source->get())}}));
      fixtures[originalSource] = source->get();
    }
    Operation *terminator = body.getTerminator();
    builder->setInsertionPoint(terminator);
    for (OpOperand &result : terminator->getOpOperands()) {
      auto type = cast<RankedTensorType>(
          graph.getResultTypes()[result.getOperandNumber()]);
      result.set(uninitialized(
          type,
          llvm::json::Object{{"kind", "result"},
                             {"index", int64_t(result.getOperandNumber())}}));
    }
    for (Operation &op : body.without_terminator())
      if (!original.contains(&op))
        kept.insert(&op);
    removeAllBut(body, kept);

    // A public graph is its own entry; a private graph uses its public caller.
    func::FuncOp caller = graph.isPrivate() ? func::FuncOp() : graph;
    module.walk([&](func::CallOp call) {
      if (graph.isPrivate() && call.getCallee() == graph.getSymName())
        caller = call->getParentOfType<func::FuncOp>();
    });
    if (!caller || caller.isPrivate())
      llvm::report_fatal_error("gvn-slice: the graph has no public caller");
    // The arguments of a private graph that the slice no longer reads are
    // dropped, then the views of the caller's arguments that fed them are
    // dead: an argument only they read is not read.
    if (graph.isPrivate()) {
      llvm::BitVector unused(graph.getNumArguments());
      for (BlockArgument argument : graph.getArguments())
        if (argument.use_empty())
          unused.set(argument.getArgNumber());
      if (failed(graph.eraseArguments(unused)))
        llvm::report_fatal_error("gvn-slice: cannot drop unread arguments");
      module.walk([&](func::CallOp call) {
        if (call.getCallee() == graph.getSymName())
          call->eraseOperands(unused);
      });
    }
    SmallVector<Operation *> ops;
    for (Operation &op : caller.front().without_terminator())
      ops.push_back(&op);
    for (Operation *op : llvm::reverse(ops))
      if (isOpTriviallyDead(op))
        op->erase();
    description["graph"] = graph.getSymName().str();
    description["entry"] = caller.getSymName().str();
    builder->setInsertionPointToEnd(module.getBody());
    auto entry = func::FuncOp::create(*builder, loc, "gvn_entry",
                                      builder->getFunctionType({}, {}));
    builder->setInsertionPointToStart(entry.addEntryBlock());
    SmallVector<Value> arguments;
    for (auto [argument, type] : llvm::enumerate(caller.getArgumentTypes())) {
      auto tensor = cast<RankedTensorType>(type);
      auto layoutAttr = caller.getArgAttrOfType<StridedLayoutAttr>(
          argument, "bufferization.buffer_layout");
      SmallVector<int64_t> identity(tensor.getRank(), 1);
      for (int64_t dim = tensor.getRank() - 2; dim >= 0; --dim)
        identity[dim] = identity[dim + 1] * tensor.getDimSize(dim + 1);
      if (caller.getArgument(argument).use_empty() && tensor.hasStaticShape()) {
        // An argument the graph does not read: a view of gvn_in, which no
        // instance allocates or frees (hundreds of empty tensors cost an
        // instance an O(n^2) run-time alias check in their deallocation).
        Value view = memref::ViewOp::create(
            *builder, loc,
            MemRefType::get(tensor.getShape(), tensor.getElementType()),
            global("gvn_in", builder->getI8Type()),
            arith::ConstantIndexOp::create(*builder, loc, 0), ValueRange{});
        arguments.push_back(bufferization::ToTensorOp::create(
            *builder, loc, tensor, view, true));
      } else if (layoutAttr && tensor.hasStaticShape() &&
                 layoutAttr.getStrides() != ArrayRef<int64_t>(identity)) {
        // The argument is a view of a larger value (a few heads of a
        // cache): gvn_in holds that container, of the sizes its strides
        // give, and the argument is its leading view.
        ArrayRef<int64_t> strides = layoutAttr.getStrides();
        SmallVector<int64_t> container{tensor.getDimSize(0)};
        for (int64_t dim = 1; dim < tensor.getRank(); ++dim) {
          if (strides[dim - 1] % strides[dim] ||
              strides[dim - 1] / strides[dim] < tensor.getDimSize(dim) ||
              strides.back() != 1)
            llvm::report_fatal_error("gvn-slice: an argument layout is no "
                                     "view of a row-major container");
          container.push_back(strides[dim - 1] / strides[dim]);
        }
        auto containerType =
            RankedTensorType::get(container, tensor.getElementType());
        int64_t slot = slots++;
        Value view = memref::ViewOp::create(
            *builder, loc, MemRefType::get(container, tensor.getElementType()),
            global("gvn_in", builder->getI8Type()), tableWord(slot),
            ValueRange{});
        SmallVector<OpFoldResult> zeros(tensor.getRank(),
                                        builder->getIndexAttr(0));
        SmallVector<OpFoldResult> ones(tensor.getRank(),
                                       builder->getIndexAttr(1));
        SmallVector<OpFoldResult> sizes;
        for (int64_t size : tensor.getShape())
          sizes.push_back(builder->getIndexAttr(size));
        Value part =
            memref::SubViewOp::create(*builder, loc, view, zeros, sizes, ones);
        entries.push_back(llvm::json::Object{
            {"kind", "argument"},
            {"index", int64_t(argument)},
            {"slot", slot},
            {"type", typeName(tensor)},
            {"container", typeName(containerType)},
            {"element", elementName(tensor.getElementType())}});
        arguments.push_back(bufferization::ToTensorOp::create(
            *builder, loc, tensor, part, true));
      } else {
        arguments.push_back(readInput(
            tensor, llvm::json::Object{{"kind", "argument"},
                                       {"index", int64_t(argument)}}));
      }
    }
    func::CallOp::create(*builder, loc, caller, arguments);
    func::ReturnOp::create(*builder, loc);
  }
};
} // namespace

namespace gvn {
void registerSlicePass() { PassRegistration<SlicePass>(); }
} // namespace gvn

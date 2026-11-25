//===- LowerDevice.cpp - A placed program on the device runtime -----------===//
//
// lower-device lowers the bufferized launches and transfers of a program,
// placed by the design's convert-linalg-to-device, to a static program for
// the program runtime (stack/runtime/program/program.h) and the device
// runtime of the design.
//
// Every launch of a graph and every copy of a value a launch wrote into a
// result is an entry, numbered in program order. What the entries are and how
// they depend on each other is fixed at compile time, in the program's table
// (<program>_table, read-only): per entry its hart, its words, how many
// entries it waits for, the entries its completion wakes, and the transfers
// it pulls at its front and pushes at its tail. A transfer of a value a launch
// writes and only entries read is no entry of its own: its producer pushes it
// at its tail, the first entry that reads it on the receiving hart pulls it at
// its front. How a value crosses (the device SHM, a copy in the DDR) is the
// design's. The program, <program>_job, only fills in what depends on the
// sizes of the run: it writes the words of every entry (the address and the
// dynamic sizes of every buffer, and of a transfer its source, target, byte
// count and logical value identity) into <program>_words and dispatches
// entries:
//
//   program_start(table, words, state)   first
//   program_dispatch(entry, job)         once the words of the entry and of
//                                        the transfers it pushes are written
//   program_wait(entry), device_copy(from, to, target, source, bytes)
//                                        a transfer of a value the program
//                                        made or reads itself, now
//   program_finish()                     last: every entry has retired; the
//                                        program frees its buffers
//
// The table, as 64-bit words: the entry count E, the transfer count T, the
// hart count C; for each hart its first entry (E if none); for each entry
// PROGRAM_ENTRY_WORDS words (hart, first word, dependency count, next entry
// of its hart, and offset and count into the lists of its completion
// wakeups, its stream wakeups, its pulls and its pushes); for each transfer
// its from hart, its to hart, its first word; then the lists. A stream
// wakeup is an entry that waits for the entry only for a value it pushes:
// the design may wake it as soon as the pushes start (program_release). The
// state, <program>_state, has PROGRAM_STATE_WORDS zero words per entry for
// the runtime.
//
// Every buffer a job reads or writes is contiguous, but one a graph takes
// with static strides (a rank's slice of a state, the heads of every token),
// which the job passes from its first element; every buffer a launch or
// a transfer writes is a buffer of its own (an allocation or an argument of
// the program), so no entry writes a buffer another entry reads before.
//
//===----------------------------------------------------------------------===//

#include "Device/DeviceOps.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace ::buddy::device;

// The words of the state of an entry (program.h).
static constexpr int64_t PROGRAM_STATE_WORDS = 5;

namespace {
class LowerDevicePass
    : public PassWrapper<LowerDevicePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerDevicePass)

  StringRef getArgument() const final { return "lower-device"; }
  StringRef getDescription() const final {
    return "Lower the launches and transfers of a placed program to the job "
           "runtime and the device runtime of the design";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    LLVMTypeConverter converter(context);
    Type i32 = IntegerType::get(context, 32);
    Type i64 = IntegerType::get(context, 64);
    Type ptr = LLVM::LLVMPointerType::get(context);
    FunctionType entryType = FunctionType::get(context, {ptr, ptr}, {i32});

    auto declare = [&](StringRef name, TypeRange arguments, TypeRange results) {
      if (auto function = module.lookupSymbol<func::FuncOp>(name))
        return function;
      OpBuilder builder = OpBuilder::atBlockEnd(module.getBody());
      auto function =
          func::FuncOp::create(builder, module.getLoc(), name,
                               FunctionType::get(context, arguments, results));
      function.setPrivate();
      return function;
    };
    func::FuncOp start = declare("program_start", {ptr, ptr, ptr}, {});
    func::FuncOp dispatch = declare("program_dispatch", {i64, entryType}, {});
    func::FuncOp wait = declare("program_wait", {i64}, {});
    func::FuncOp finish = declare("program_finish", {}, {});
    func::FuncOp copyJob = declare("program_copy_job", {ptr, ptr}, {i32});
    func::FuncOp transferJob;
    func::FuncOp move = declare("device_copy", {i64, i64, i64, i64, i64}, {});

    // <function>_job(hart context, words): for every memref argument of
    // `function` its address and its dynamic sizes, rebuilt into the memref
    // with the identity layout; then the call.
    auto entry = [&](func::FuncOp function) {
      std::string name = (function.getName() + "_job").str();
      if (auto existing = module.lookupSymbol<func::FuncOp>(name))
        return existing;
      // A graph is compiled with its results as buffers after its inputs
      // (buffer-results-to-out-params), which leaves its declaration here
      // as it was.
      if (function.isExternal() && function.getNumResults()) {
        SmallVector<Type> arguments(function.getArgumentTypes());
        llvm::append_range(arguments, function.getResultTypes());
        function.setType(FunctionType::get(context, arguments, {}));
      }
      OpBuilder builder = OpBuilder::atBlockEnd(module.getBody());
      Location loc = function.getLoc();
      auto job = func::FuncOp::create(builder, loc, name, entryType);
      builder.setInsertionPointToStart(job.addEntryBlock());
      int32_t word = 0;
      auto next = [&]() {
        Value slot =
            LLVM::GEPOp::create(builder, loc, ptr, ptr, job.getArgument(1),
                                ArrayRef<LLVM::GEPArg>{word++});
        return LLVM::LoadOp::create(builder, loc, ptr, slot).getResult();
      };
      SmallVector<Value> arguments;
      for (Type type : function.getArgumentTypes()) {
        auto memref = cast<MemRefType>(type);
        Value pointer = next();
        Value descriptor;
        if (!memref.getLayout().isIdentity()) {
          // Static sizes and strides from the first element.
          auto [strides, offset] = memref.getStridesAndOffset();
          if (!memref.hasStaticShape() ||
              llvm::any_of(strides, ShapedType::isDynamic))
            llvm::report_fatal_error(llvm::Twine("argument of ") +
                                     function.getName() +
                                     " with dynamic strides");
          auto built = MemRefDescriptor::poison(builder, loc,
                                                converter.convertType(memref));
          built.setAllocatedPtr(builder, loc, pointer);
          built.setAlignedPtr(builder, loc, pointer);
          built.setConstantOffset(builder, loc, 0);
          for (int64_t axis = 0; axis < memref.getRank(); ++axis) {
            built.setConstantSize(builder, loc, axis, memref.getDimSize(axis));
            built.setConstantStride(builder, loc, axis, strides[axis]);
          }
          descriptor = built;
        } else if (memref.hasStaticShape()) {
          descriptor = MemRefDescriptor::fromStaticShape(
              builder, loc, converter, memref, pointer);
        } else {
          // Row-major strides over the sizes, the dynamic ones from the
          // words.
          auto built = MemRefDescriptor::poison(builder, loc,
                                                converter.convertType(memref));
          built.setAllocatedPtr(builder, loc, pointer);
          built.setAlignedPtr(builder, loc, pointer);
          built.setConstantOffset(builder, loc, 0);
          SmallVector<Value> sizes;
          for (int64_t size : memref.getShape())
            sizes.push_back(
                ShapedType::isDynamic(size)
                    ? LLVM::PtrToIntOp::create(builder, loc, i64, next())
                          .getResult()
                    : LLVM::ConstantOp::create(builder, loc, i64, size)
                          .getResult());
          Value stride = LLVM::ConstantOp::create(builder, loc, i64, 1);
          for (int64_t axis = memref.getRank() - 1; axis >= 0; --axis) {
            built.setSize(builder, loc, axis, sizes[axis]);
            built.setStride(builder, loc, axis, stride);
            stride = LLVM::MulOp::create(builder, loc, stride, sizes[axis]);
          }
          descriptor = built;
        }
        arguments.push_back(
            UnrealizedConversionCastOp::create(builder, loc, memref, descriptor)
                .getResult(0));
      }
      func::CallOp::create(builder, loc, function, arguments);
      func::ReturnOp::create(
          builder, loc,
          ValueRange{arith::ConstantIntOp::create(builder, loc, 0, 32)});
      return job;
    };

    // The buffers a graph takes with their strides (a strided argument).
    DenseSet<Value> strided;
    auto contiguous = [&](Value buffer) {
      auto type = cast<MemRefType>(buffer.getType());
      if (type.getLayout().isIdentity() || strided.contains(buffer))
        return;
      auto [strides, offset] = type.getStridesAndOffset();
      int64_t expected = 1;
      for (int64_t axis = type.getRank() - 1; axis >= 0; --axis) {
        int64_t size = type.getDimSize(axis);
        if (size == 1)
          continue;
        if (ShapedType::isDynamic(expected) || strides[axis] != expected)
          llvm::report_fatal_error(
              "a job reads or writes a buffer that is not contiguous");
        expected = ShapedType::isDynamic(size) ? ShapedType::kDynamic
                                               : expected * size;
      }
    };
    // The address of the first element of a buffer.
    auto address = [&](OpBuilder &builder, Location loc, Value buffer) {
      contiguous(buffer);
      auto type = cast<MemRefType>(buffer.getType());
      Value base =
          memref::ExtractAlignedPointerAsIndexOp::create(builder, loc, buffer);
      Value offset =
          memref::ExtractStridedMetadataOp::create(builder, loc, buffer)
              .getOffset();
      Value bytes = arith::ConstantIndexOp::create(
          builder, loc, (type.getElementTypeBitWidth() + 7) / 8);
      Value first = arith::AddIOp::create(
          builder, loc, base,
          arith::MulIOp::create(builder, loc, offset, bytes));
      return arith::IndexCastOp::create(builder, loc, i64, first).getResult();
    };
    auto constant = [&](OpBuilder &builder, Location loc, int64_t value) {
      return arith::ConstantIntOp::create(builder, loc, value, 64).getResult();
    };
    // The dynamic sizes of a buffer.
    auto sizes = [&](OpBuilder &builder, Location loc, Value buffer) {
      SmallVector<Value> result;
      for (auto [axis, size] :
           llvm::enumerate(cast<MemRefType>(buffer.getType()).getShape()))
        if (ShapedType::isDynamic(size))
          result.push_back(arith::IndexCastOp::create(
              builder, loc, i64,
              memref::DimOp::create(builder, loc, buffer, axis)));
      return result;
    };
    // The bytes of a buffer.
    auto bytes = [&](OpBuilder &builder, Location loc, Value buffer) {
      auto type = cast<MemRefType>(buffer.getType());
      Value total =
          constant(builder, loc,
                   (type.getElementTypeBitWidth() + 7) / 8 *
                       (type.hasStaticShape() ? type.getNumElements() : 1));
      if (!type.hasStaticShape()) {
        for (auto [axis, size] : llvm::enumerate(type.getShape()))
          total = arith::MulIOp::create(
              builder, loc, total,
              ShapedType::isDynamic(size)
                  ? arith::IndexCastOp::create(
                        builder, loc, i64,
                        memref::DimOp::create(builder, loc, buffer, axis))
                        .getResult()
                  : constant(builder, loc, size));
      }
      return total;
    };
    // The buffer a view is of.
    auto root = [](Value buffer) {
      while (auto view =
                 dyn_cast_or_null<ViewLikeOpInterface>(buffer.getDefiningOp()))
        buffer = view.getViewSource();
      return buffer;
    };
    // A buffer each launch and transfer writes: its own.
    auto own = [&](Value buffer) {
      buffer = root(buffer);
      if (!isa<BlockArgument>(buffer) &&
          !buffer.getDefiningOp<memref::AllocOp>())
        llvm::report_fatal_error("a job writes a buffer that is not its own");
      return buffer;
    };
    // Whether the program itself reads `buffer`: an operation other than a
    // launch, a transfer or a copy (which jobs do), through its views too.
    std::function<bool(Value, Operation *)> read =
        [&](Value buffer, Operation *writer) -> bool {
      for (Operation *user : buffer.getUsers()) {
        if (user == writer || isa<LaunchOp, TransferOp, memref::CopyOp>(user) ||
            isMemoryEffectFree(user))
          continue;
        if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
          if (read(view->getResult(0), writer))
            return true;
          continue;
        }
        return true;
      }
      return false;
    };

    SmallVector<func::FuncOp> programs;
    for (auto function : module.getOps<func::FuncOp>())
      if (!function.isExternal() && !function.getOps<LaunchOp>().empty())
        programs.push_back(function);
    for (func::FuncOp program : programs) {
      uint64_t control =
          program->getAttrOfType<IntegerAttr>("device.hart").getInt();
      program->removeAttr("device.hart");
      std::string words = (program.getName() + "_words").str();
      std::string state = (program.getName() + "_state").str();
      std::string table = (program.getName() + "_table").str();

      struct Entry {
        int64_t hart;
        int64_t words;
        SetVector<int64_t> dependencies, streams;
        SmallVector<int64_t> pulls, pushes;
      };
      struct Transfer {
        int64_t from, to, words, producer, consumer = -1;
        // The buffer it moves into, once the program placed it.
        Value target;
      };
      SmallVector<Entry> entries;
      SmallVector<Transfer> transfers;
      DenseMap<Value, int64_t> valueIdentities;
      auto identity = [&](Value value) {
        return valueIdentities.try_emplace(value, valueIdentities.size())
            .first->second;
      };
      int64_t wordCount = 0;
      // The entries that write into each buffer, through any of its views,
      // and the transfer its first reader pulls.
      DenseMap<Value, SmallVector<int64_t>> writers;
      DenseMap<Value, int64_t> pulled;
      // The asynchronous transfers of each source: the value a launch
      // writes, which only entries read.
      DenseMap<Value, SmallVector<TransferOp>> sources;
      DenseMap<Value, TransferOp> fanoutLast;
      DenseMap<Value, int64_t> fanoutEntries;
      DenseMap<Operation *, int64_t> indices;
      program.walk([&](TransferOp moved) {
        if (!moved->hasAttr("device.defer") && !read(moved.getTarget(), moved))
          sources[moved.getSource()].push_back(moved);
        if (moved->hasAttr("device.fanout"))
          fanoutLast[moved.getSource()] = moved;
      });

      OpBuilder top = OpBuilder::atBlockBegin(&program.getBody().front());
      Location programLoc = program.getLoc();
      auto pointer = [&](OpBuilder &builder, Location loc, StringRef name) {
        return LLVM::AddressOfOp::create(builder, loc, ptr, name).getResult();
      };
      Value wordsPointer = pointer(top, programLoc, words);
      func::CallOp::create(top, programLoc, start,
                           ValueRange{pointer(top, programLoc, table),
                                      wordsPointer,
                                      pointer(top, programLoc, state)});
      auto store = [&](OpBuilder &builder, Location loc, int64_t index,
                       Value value) {
        Value slot =
            LLVM::GEPOp::create(builder, loc, ptr, i64, wordsPointer,
                                ArrayRef<LLVM::GEPArg>{int32_t(index)});
        LLVM::StoreOp::create(builder, loc, value, slot);
      };
      // The entries `reader` waits for to read `buffer` on `hart`; a
      // transfer not pulled yet is pulled by it, one an entry of another
      // hart pulled is there once that entry ran.
      auto depend = [&](Entry &reader, int64_t index, Value buffer) {
        if (auto found = pulled.find(buffer); found != pulled.end()) {
          Transfer &moved = transfers[found->second];
          if (moved.consumer < 0) {
            moved.consumer = index;
            reader.pulls.push_back(found->second);
          } else if (entries[moved.consumer].hart != reader.hart) {
            reader.dependencies.insert(moved.consumer);
          }
          return;
        }
        for (int64_t writer : writers.lookup(root(buffer)))
          if (entries[writer].hart != reader.hart)
            reader.dependencies.insert(writer);
      };
      // An entry on `hart` that reads `inputs`, with the words of `buffers`.
      auto add = [&](OpBuilder &builder, Location loc, int64_t hart,
                     ValueRange inputs, ValueRange buffers) {
        int64_t index = entries.size();
        entries.push_back({hart, wordCount, {}, {}, {}, {}});
        for (Value input : inputs)
          depend(entries[index], index, input);
        for (Value buffer : buffers) {
          store(builder, loc, wordCount++, address(builder, loc, buffer));
          for (Value size : sizes(builder, loc, buffer))
            store(builder, loc, wordCount++, size);
        }
        return index;
      };
      // The transfers of what entry `index` writes into `out`: their
      // sources and bytes now, before the entry is dispatched; their
      // targets where they are, before their readers are.
      auto push = [&](OpBuilder &builder, Location loc, int64_t index,
                      Value out) {
        for (TransferOp moved : sources.lookup(out)) {
          own(moved.getTarget());
          int64_t t = transfers.size();
          transfers.push_back({int64_t(moved.getFrom()), int64_t(moved.getTo()),
                               wordCount, index});
          entries[index].pushes.push_back(t);
          store(builder, loc, wordCount, address(builder, loc, out));
          store(builder, loc, wordCount + 2, bytes(builder, loc, out));
          store(builder, loc, wordCount + 3,
                arith::ConstantIntOp::create(builder, loc, identity(out), 64));
          wordCount += 4;
          indices[moved] = t;
        }
      };

      for (Operation &op : llvm::make_early_inc_range(program.getOps())) {
        OpBuilder builder(&op);
        Location loc = op.getLoc();
        if (auto launch = dyn_cast<LaunchOp>(op)) {
          auto callee = module.lookupSymbol<func::FuncOp>(launch.getCallee());
          // Its results are buffers after its inputs (entry).
          SmallVector<Type> types(callee.getArgumentTypes());
          llvm::append_range(types, callee.getResultTypes());
          for (auto [operand, type] :
               llvm::zip_equal(launch->getOperands(), types)) {
            auto expected = cast<MemRefType>(type);
            if (expected.getLayout().isIdentity())
              continue;
            if (cast<MemRefType>(operand.getType())
                    .getStridesAndOffset()
                    .first != expected.getStridesAndOffset().first)
              llvm::report_fatal_error(
                  "a launch passes a buffer of other strides "
                  "than its graph takes");
            strided.insert(operand);
          }
          Value job = func::ConstantOp::create(builder, loc, entryType,
                                               entry(callee).getName());
          int64_t index = add(builder, loc, *launch.getHart(),
                              launch.getInputs(), launch->getOperands());
          for (Value out : launch.getOuts()) {
            writers[own(out)].push_back(index);
            push(builder, loc, index, out);
          }
          func::CallOp::create(builder, loc, dispatch,
                               ValueRange{constant(builder, loc, index), job});
          launch.erase();
          continue;
        }
        if (auto moved = dyn_cast<TransferOp>(op)) {
          if (moved->hasAttr("device.defer")) {
            if (!transferJob) {
              OpBuilder definition = OpBuilder::atBlockEnd(module.getBody());
              transferJob = func::FuncOp::create(
                  definition, loc, "program_transfer_job", entryType);
              transferJob.setPrivate();
              definition.setInsertionPointToStart(transferJob.addEntryBlock());
              Value success =
                  arith::ConstantIntOp::create(definition, loc, 0, 32);
              func::ReturnOp::create(definition, loc, success);
            }
            own(moved.getTarget());
            int64_t index;
            auto group = fanoutEntries.find(moved.getSource());
            if (moved->hasAttr("device.fanout") &&
                group != fanoutEntries.end()) {
              index = group->second;
            } else {
              index = add(builder, loc, moved.getFrom(),
                          ValueRange{moved.getSource()}, ValueRange{});
              if (moved->hasAttr("device.fanout"))
                fanoutEntries[moved.getSource()] = index;
            }
            int64_t transfer = transfers.size();
            transfers.push_back({int64_t(moved.getFrom()),
                                 int64_t(moved.getTo()), wordCount, index, -1,
                                 moved.getTarget()});
            entries[index].pushes.push_back(transfer);
            store(builder, loc, wordCount++,
                  address(builder, loc, moved.getSource()));
            store(builder, loc, wordCount++,
                  address(builder, loc, moved.getTarget()));
            store(builder, loc, wordCount++,
                  bytes(builder, loc, moved.getSource()));
            store(builder, loc, wordCount++,
                  arith::ConstantIntOp::create(
                      builder, loc, identity(moved.getSource()), 64));
            pulled[moved.getTarget()] = transfer;
            if (!moved->hasAttr("device.fanout") ||
                fanoutLast.lookup(moved.getSource()) == moved) {
              Value job = func::ConstantOp::create(builder, loc, entryType,
                                                   transferJob.getName());
              func::CallOp::create(
                  builder, loc, dispatch,
                  ValueRange{constant(builder, loc, index), job});
            }
            moved.erase();
            continue;
          }
          if (auto found = indices.find(moved); found != indices.end()) {
            // Asynchronous: the target, which its first reader pulls.
            Transfer &t = transfers[found->second];
            store(builder, loc, t.words + 1,
                  address(builder, loc, moved.getTarget()));
            t.target = moved.getTarget();
            pulled[moved.getTarget()] = found->second;
            moved.erase();
            continue;
          }
          if (pulled.contains(moved.getSource()))
            llvm::report_fatal_error(
                "the program moves a value a transfer moved");
          for (int64_t writer : writers.lookup(root(moved.getSource())))
            func::CallOp::create(builder, loc, wait,
                                 ValueRange{constant(builder, loc, writer)});
          func::CallOp::create(
              builder, loc, move,
              ValueRange{constant(builder, loc, moved.getFrom()),
                         constant(builder, loc, moved.getTo()),
                         address(builder, loc, moved.getTarget()),
                         address(builder, loc, moved.getSource()),
                         bytes(builder, loc, moved.getSource())});
          moved.erase();
          continue;
        }
        if (auto view = dyn_cast<ViewLikeOpInterface>(op)) {
          if (auto found = pulled.find(view.getViewSource());
              found != pulled.end())
            pulled[op.getResult(0)] = found->second;
          continue;
        }
        if (auto copied = dyn_cast<memref::CopyOp>(op)) {
          if (copied.getSource() == copied.getTarget()) {
            copied.erase();
            continue;
          }
          if (!writers.contains(root(copied.getSource())) &&
              !pulled.contains(copied.getSource()))
            continue;
          Value target = own(copied.getTarget());
          Value job = func::ConstantOp::create(builder, loc, entryType,
                                               copyJob.getName());
          int64_t index = add(builder, loc, control,
                              ValueRange{copied.getSource()}, ValueRange{});
          store(builder, loc, wordCount++,
                address(builder, loc, copied.getTarget()));
          store(builder, loc, wordCount++,
                address(builder, loc, copied.getSource()));
          store(builder, loc, wordCount++,
                bytes(builder, loc, copied.getSource()));
          writers[target].push_back(index);
          func::CallOp::create(builder, loc, dispatch,
                               ValueRange{constant(builder, loc, index), job});
          copied.erase();
          continue;
        }
        // Sizes and other metadata of a buffer do not read it.
        if (isMemoryEffectFree(&op))
          continue;
        for (Value operand : op.getOperands())
          if (writers.contains(root(operand)) || pulled.contains(operand))
            llvm::report_fatal_error(
                llvm::Twine("the program reads a value of an entry in ") +
                op.getName().getStringRef());
      }
      // A transfer into a result of the program that no entry reads: an
      // empty copy entry on its target hart pulls it before the program
      // returns.
      {
        OpBuilder builder(program.getBody().front().getTerminator());
        Location loc = program.getLoc();
        SmallVector<std::pair<int64_t, Value>> unread;
        for (auto [target, t] : pulled)
          if (transfers[t].consumer < 0 && target == transfers[t].target)
            unread.push_back({t, target});
        llvm::sort(unread, llvm::less_first());
        for (auto [t, target] : unread) {
          Value job = func::ConstantOp::create(builder, loc, entryType,
                                               copyJob.getName());
          int64_t index =
              add(builder, loc, transfers[t].to, ValueRange{target}, {});
          store(builder, loc, wordCount++, address(builder, loc, target));
          store(builder, loc, wordCount++, address(builder, loc, target));
          store(builder, loc, wordCount++, constant(builder, loc, 0));
          func::CallOp::create(builder, loc, dispatch,
                               ValueRange{constant(builder, loc, index), job});
        }
      }
      for (const Transfer &t : transfers)
        if (t.consumer < 0)
          llvm::report_fatal_error("a transfer that no entry reads");
      // Every entry has retired: the buffers are free.
      {
        OpBuilder builder(program.getBody().front().getTerminator());
        func::CallOp::create(builder, programLoc, finish, ValueRange{});
        for (auto allocation : program.getOps<memref::AllocOp>())
          memref::DeallocOp::create(builder, programLoc,
                                    allocation.getResult());
      }

      // The table.
      int64_t entryCount = entries.size(), hartCount = 0;
      for (const Entry &e : entries)
        hartCount = std::max(hartCount, e.hart + 1);
      SmallVector<int64_t> lists, first(hartCount, entryCount),
          next(entryCount, entryCount);
      SmallVector<int64_t> last(hartCount, -1);
      for (auto [index, e] : llvm::enumerate(entries)) {
        if (last[e.hart] < 0)
          first[e.hart] = index;
        else
          next[last[e.hart]] = index;
        last[e.hart] = index;
      }
      // The first sender starts the reader. Each pull waits for its own
      // header; waiting for later sends here can block their input broadcasts.
      for (auto [index, e] : llvm::enumerate(entries)) {
        llvm::sort(e.pulls, [&](int64_t a, int64_t b) {
          return std::make_pair(transfers[a].producer, a) <
                 std::make_pair(transfers[b].producer, b);
        });
        if (!e.pulls.empty())
          entries[transfers[e.pulls.front()].producer].streams.insert(index);
      }
      SmallVector<SmallVector<int64_t>> wakeups(entryCount),
          streams(entryCount);
      SmallVector<int64_t> waits(entryCount, 0);
      for (auto [index, e] : llvm::enumerate(entries)) {
        for (int64_t producer : e.dependencies) {
          wakeups[producer].push_back(index);
          ++waits[index];
        }
      }
      for (auto [index, e] : llvm::enumerate(entries))
        for (int64_t consumer : e.streams) {
          streams[index].push_back(consumer);
          ++waits[consumer];
        }
      SmallVector<int64_t> data{entryCount, int64_t(transfers.size()),
                                hartCount};
      llvm::append_range(data, first);
      auto list = [&](ArrayRef<int64_t> values) {
        data.push_back(lists.size());
        data.push_back(values.size());
        llvm::append_range(lists, values);
      };
      for (auto [index, e] : llvm::enumerate(entries)) {
        data.append({e.hart, e.words, waits[index], next[index]});
        list(wakeups[index]);
        list(streams[index]);
        list(e.pulls);
        list(e.pushes);
      }
      for (const Transfer &t : transfers)
        data.append({t.from, t.to, t.words});
      llvm::append_range(data, lists);
      OpBuilder global = OpBuilder::atBlockEnd(module.getBody());
      auto array = [&](int64_t count) {
        return LLVM::LLVMArrayType::get(i64, std::max<int64_t>(count, 1));
      };
      auto values = [&](ArrayRef<int64_t> content, int64_t count) {
        SmallVector<int64_t> padded(content);
        padded.resize(std::max<int64_t>(count, 1), 0);
        return DenseElementsAttr::get(
            RankedTensorType::get({int64_t(padded.size())}, i64),
            ArrayRef<int64_t>(padded));
      };
      LLVM::GlobalOp::create(global, programLoc, array(data.size()), true,
                             LLVM::Linkage::Internal, table,
                             values(data, data.size()));
      LLVM::GlobalOp::create(global, programLoc, array(wordCount), false,
                             LLVM::Linkage::Internal, words,
                             values({}, wordCount));
      LLVM::GlobalOp::create(global, programLoc,
                             array(entryCount * PROGRAM_STATE_WORDS), false,
                             LLVM::Linkage::Internal, state,
                             values({}, entryCount * PROGRAM_STATE_WORDS));
      entry(program);
    }
  }
};

PassRegistration<LowerDevicePass> registration;
} // namespace

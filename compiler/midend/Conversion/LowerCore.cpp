//===- LowerCore.cpp - The subgraphs of a graph on the engine -------------===//
//
// lower-core lowers the bufferized launches and transfers of a graph, placed
// by the design's convert-linalg-to-core and after its deallocation, to the
// core CPU and asynchronous jobs of the engine
// (stack/engine/include/stream.h):
//
//   transfer   no data moves on the CPU: a thread reads and writes DDR like
//              the CPU, so the launches read and write the buffers of the
//              values themselves, in their layout (rows that follow each
//              other, from the first element), and the transfer keeps only
//              its order. Between two threads, stream_transfer(source job,
//              target job, address, address, bytes) lets the core pass the
//              value from one thread to the other (engine_transfer)
//   launch     call <callee>_launch(buffers): the pointer of every buffer
//              followed by its dynamic sizes in an array, then
//              stream_job(signature and index of the thread,
//              <callee>_thread, array), the next job of the open batch, and
//              stream_after(job) for each earlier job of the batch that
//              writes a buffer it reads or writes, or reads one it writes,
//              and that does not pass it a value already
//
// The launches of a graph form batches, in order: a batch takes the launches
// up to the first operation of the CPU that touches a buffer one of them
// writes, or writes or frees a buffer one of them reads (views of a buffer
// are the buffer), or the end of the block.
// stream_begin(jobs, dependencies, transfers) comes before its first launch,
// stream_submit() after its last, and stream_wait() before that operation:
// the threads run the batch while the CPU goes on, each job on the thread
// placed for it once its dependencies and transfers are done, so the jobs of
// a batch on different threads run at the same time; the CPU reads a result
// only after its job.
//
// stream_job runs <callee>_thread on the core CPU, which rebuilds the memrefs
// of the subgraph from the pointers and calls it; the Ball instructions the
// subgraph issues are the job's stream, which the thread executes once the
// batch is submitted. The signature is the thread's in design.pb, the same
// on every core of the class (`core`). A subgraph takes buffers of the
// identity layout: a buffer of another layout whose rows follow each other
// is passed as such a buffer from its first element, so a pointer and its
// dynamic sizes are the whole buffer.
//
//===----------------------------------------------------------------------===//

#include "Core/CoreOps.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/MemoryBuffer.h"

#include "design.pb.h"

using namespace mlir;
using namespace ::buddy::core;

namespace mlir::buddy {
LogicalResult lowerThreadStream(func::FuncOp function);
}

namespace {
class LowerCorePass
    : public PassWrapper<LowerCorePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCorePass)

  LowerCorePass() = default;
  LowerCorePass(const LowerCorePass &pass) : PassWrapper(pass) {}

  StringRef getArgument() const final { return "lower-core"; }
  StringRef getDescription() const final {
    return "Lower the launches and transfers of a graph to DDR buffers and "
           "the engine of its core";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect,
                    memref::MemRefDialect>();
  }

  Option<std::string> designPb{*this, "design",
                               llvm::cl::desc("design.pb of the device")};
  Option<std::string> coreClass{
      *this, "core", llvm::cl::desc("class of the core the graph runs on")};

  void runOnOperation() override {
    auto buffer = llvm::MemoryBuffer::getFile(designPb);
    if (!buffer)
      llvm::report_fatal_error(llvm::Twine("cannot read ") + designPb);
    buckyball::config::Design design;
    if (!design.ParseFromArray((*buffer)->getBufferStart(),
                               (*buffer)->getBufferSize()))
      llvm::report_fatal_error(llvm::Twine("cannot parse ") + designPb);
    // The signature of each thread of the core class, by its index in the
    // core.
    SmallVector<uint64_t> signatures;
    for (const auto &core : design.device().cores()) {
      if (core.component() != coreClass)
        continue;
      SmallVector<uint64_t> threads;
      for (uint32_t index : core.thread_indices())
        threads.push_back(design.threads(index).signature());
      if (!signatures.empty() && signatures != threads)
        llvm::report_fatal_error(llvm::Twine("the cores of class ") +
                                 coreClass + " differ in their threads");
      signatures = threads;
    }
    if (signatures.empty())
      llvm::report_fatal_error(llvm::Twine("no core of class ") + coreClass +
                               " has a thread");

    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    LLVMTypeConverter converter(context);
    Type i64 = IntegerType::get(context, 64);
    Type ptr = LLVM::LLVMPointerType::get(context);
    FunctionType entryType = FunctionType::get(context, {ptr}, {});

    auto declare = [&](StringRef name, TypeRange arguments) {
      if (auto function = module.lookupSymbol<func::FuncOp>(name))
        return function;
      OpBuilder builder = OpBuilder::atBlockEnd(module.getBody());
      auto function =
          func::FuncOp::create(builder, module.getLoc(), name,
                               FunctionType::get(context, arguments, {}));
      function.setPrivate();
      return function;
    };
    Type i32 = IntegerType::get(context, 32);
    func::FuncOp begin = declare("stream_begin", {i64, i64, i64});
    func::FuncOp job = declare("stream_job", {i64, i32, entryType, ptr});
    func::FuncOp after = declare("stream_after", {i64});
    // stream_transfer only where values pass between two threads.
    bool between = false;
    module.walk(
        [&](TransferOp move) { between |= move.getFrom() && move.getTo(); });
    func::FuncOp pass =
        between ? declare("stream_transfer", {i64, i64, ptr, ptr, i64})
                : func::FuncOp();
    func::FuncOp submit = declare("stream_submit", {});
    func::FuncOp wait = declare("stream_wait", {});

    // Each outlined body records its Ball commands for its hardware thread.
    DenseSet<Operation *> recorded;
    bool invalidStream = false;
    module.walk([&](LaunchOp launch) {
      auto callee = module.lookupSymbol<func::FuncOp>(launch.getCallee());
      if (recorded.insert(callee).second &&
          failed(mlir::buddy::lowerThreadStream(callee)))
        invalidStream = true;
    });
    if (invalidStream)
      return signalPassFailure();

    // <callee>_thread(words): the memrefs of `callee` rebuilt from the
    // pointers and the dynamic sizes, then the call.
    auto entry = [&](func::FuncOp callee) {
      OpBuilder builder = OpBuilder::atBlockEnd(module.getBody());
      Location loc = callee.getLoc();
      auto function = func::FuncOp::create(
          builder, loc, (callee.getName() + "_thread").str(), entryType);
      function.setPrivate();
      builder.setInsertionPointToStart(function.addEntryBlock());
      auto word = [&](int64_t index, Type type) -> Value {
        Value slot = LLVM::GEPOp::create(
            builder, loc, ptr, ptr, function.getArgument(0),
            ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(index)});
        return LLVM::LoadOp::create(builder, loc, type, slot);
      };
      SmallVector<Value> arguments;
      int64_t next = 0;
      for (Type type : callee.getArgumentTypes()) {
        auto memref = cast<MemRefType>(type);
        if (!memref.getLayout().isIdentity())
          llvm::report_fatal_error(llvm::Twine("argument of ") +
                                   callee.getName() +
                                   " without the identity layout");
        Value pointer = word(next++, ptr);
        Value descriptor;
        if (memref.hasStaticShape()) {
          descriptor = MemRefDescriptor::fromStaticShape(
              builder, loc, converter, memref, pointer);
        } else {
          // Row-major strides over the sizes, the dynamic ones from the
          // words.
          MemRefDescriptor built = MemRefDescriptor::poison(
              builder, loc, converter.convertType(memref));
          built.setAllocatedPtr(builder, loc, pointer);
          built.setAlignedPtr(builder, loc, pointer);
          built.setConstantOffset(builder, loc, 0);
          SmallVector<Value> sizes;
          for (int64_t size : memref.getShape())
            sizes.push_back(
                ShapedType::isDynamic(size)
                    ? word(next++, i64)
                    : LLVM::ConstantOp::create(
                          builder, loc, i64, builder.getI64IntegerAttr(size)));
          Value stride = LLVM::ConstantOp::create(builder, loc, i64,
                                                  builder.getI64IntegerAttr(1));
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
      func::CallOp::create(builder, loc, callee, arguments);
      func::ReturnOp::create(builder, loc);
      return function;
    };

    // <callee>_launch(buffers): their pointers, then stream_job.
    auto launcher = [&](func::FuncOp callee, uint64_t signature,
                        int64_t thread) {
      std::string name = (callee.getName() + "_launch").str();
      if (auto existing = module.lookupSymbol<func::FuncOp>(name))
        return existing;
      OpBuilder builder = OpBuilder::atBlockEnd(module.getBody());
      Location loc = callee.getLoc();
      auto function = func::FuncOp::create(
          builder, loc, name,
          FunctionType::get(context, callee.getArgumentTypes(), {}));
      function.setPrivate();
      builder.setInsertionPointToStart(function.addEntryBlock());
      int64_t words = callee.getNumArguments();
      for (Type type : callee.getArgumentTypes())
        words += cast<MemRefType>(type).getNumDynamicDims();
      Value count = LLVM::ConstantOp::create(builder, loc, i64,
                                             builder.getI64IntegerAttr(words));
      Value pointers = LLVM::AllocaOp::create(builder, loc, ptr, ptr, count);
      auto store = [&](int64_t index, Value value) {
        Value slot = LLVM::GEPOp::create(
            builder, loc, ptr, ptr, pointers,
            ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(index)});
        LLVM::StoreOp::create(builder, loc, value, slot);
      };
      int64_t next = 0;
      for (Value buffer : function.getArguments()) {
        Value address = memref::ExtractAlignedPointerAsIndexOp::create(
            builder, loc, buffer);
        Value word = arith::IndexCastOp::create(builder, loc, i64, address);
        store(next++, LLVM::IntToPtrOp::create(builder, loc, ptr, word));
        auto type = cast<MemRefType>(buffer.getType());
        for (int64_t axis = 0; axis < type.getRank(); ++axis)
          if (type.isDynamicDim(axis))
            store(next++,
                  arith::IndexCastOp::create(
                      builder, loc, i64,
                      memref::DimOp::create(builder, loc, buffer, axis)));
      }
      Value body = func::ConstantOp::create(builder, loc, entryType,
                                            entry(callee).getName());
      Value code = arith::ConstantIntOp::create(
          builder, loc, static_cast<int64_t>(signature), 64);
      Value index = arith::ConstantIntOp::create(builder, loc, thread, 32);
      func::CallOp::create(builder, loc, job,
                           ValueRange{code, index, body, pointers});
      func::ReturnOp::create(builder, loc);
      return function;
    };

    module.walk([&](Operation *op) {
      if (isa<LaunchOp, TransferOp>(op) &&
          !isa<func::FuncOp>(op->getParentOp())) {
        op->emitError("a launch or transfer inside a region");
        llvm::report_fatal_error("lower-core: launches and transfers "
                                 "stand in the body of a function");
      }
    });

    auto constant = [&](OpBuilder &builder, Location loc, int64_t value) {
      return arith::ConstantIntOp::create(builder, loc, value, 64).getResult();
    };
    // The address of the first element of a buffer.
    auto pointer = [&](OpBuilder &builder, Location loc, Value buffer) {
      auto type = cast<MemRefType>(buffer.getType());
      Value base =
          memref::ExtractAlignedPointerAsIndexOp::create(builder, loc, buffer);
      auto metadata =
          memref::ExtractStridedMetadataOp::create(builder, loc, buffer);
      Value offset = arith::MulIOp::create(
          builder, loc, metadata.getOffset(),
          arith::ConstantIndexOp::create(builder, loc,
                                         type.getElementTypeBitWidth() / 8));
      Value word = arith::IndexCastOp::create(
          builder, loc, i64, arith::AddIOp::create(builder, loc, base, offset));
      return LLVM::IntToPtrOp::create(builder, loc, ptr, word).getResult();
    };
    // The bytes of a buffer of the identity layout.
    auto bytes = [&](OpBuilder &builder, Location loc, Value buffer) {
      auto type = cast<MemRefType>(buffer.getType());
      Value count = arith::ConstantIndexOp::create(
          builder, loc, type.getElementTypeBitWidth() / 8);
      for (int64_t axis = 0; axis < type.getRank(); ++axis)
        count = arith::MulIOp::create(
            builder, loc, count,
            memref::DimOp::create(builder, loc, buffer, axis));
      return arith::IndexCastOp::create(builder, loc, i64, count).getResult();
    };
    // `buffer` itself as a buffer of `type`, the identity layout a thread
    // takes: its elements, row after row from its first one. A buffer of
    // another layout is in DDR already, so the thread reads it in place; its
    // rows must follow each other.
    DenseMap<Value, Value> rebased;
    auto rebase = [&](OpBuilder &builder, Location loc, Value buffer,
                      MemRefType type) -> Value {
      auto from = cast<MemRefType>(buffer.getType());
      if (from == type)
        return buffer;
      auto [strides, offset] = from.getStridesAndOffset();
      int64_t expected = 1;
      for (int64_t axis = from.getRank() - 1; axis >= 0; --axis) {
        int64_t size = from.getDimSize(axis);
        if (size != 1 && strides[axis] != expected) {
          mlir::emitError(loc, "a thread reads ") << from;
          llvm::report_fatal_error("lower-core: a thread reads a buffer "
                                   "whose rows do not follow each other");
        }
        expected =
            ShapedType::isDynamic(size) || ShapedType::isDynamic(expected)
                ? ShapedType::kDynamic
                : expected * size;
      }
      auto metadata =
          memref::ExtractStridedMetadataOp::create(builder, loc, buffer);
      MemRefDescriptor built =
          MemRefDescriptor::poison(builder, loc, converter.convertType(type));
      Value address = pointer(builder, loc, buffer);
      built.setAllocatedPtr(builder, loc, address);
      built.setAlignedPtr(builder, loc, address);
      built.setConstantOffset(builder, loc, 0);
      Value stride = LLVM::ConstantOp::create(builder, loc, i64,
                                              builder.getI64IntegerAttr(1));
      for (int64_t axis = type.getRank() - 1; axis >= 0; --axis) {
        Value size = arith::IndexCastOp::create(builder, loc, i64,
                                                metadata.getSizes()[axis]);
        built.setSize(builder, loc, axis, size);
        built.setStride(builder, loc, axis, stride);
        stride = LLVM::MulOp::create(builder, loc, stride, size);
      }
      Value view =
          UnrealizedConversionCastOp::create(builder, loc, type, Value(built))
              .getResult(0);
      rebased[view] = buffer;
      return view;
    };
    // The buffer a view is of.
    std::function<Value(Value)> root = [&](Value value) -> Value {
      if (auto found = rebased.find(value); found != rebased.end())
        return root(found->second);
      if (auto view = value.getDefiningOp<ViewLikeOpInterface>())
        return root(view.getViewSource());
      return value;
    };
    // Erases the deallocations of a buffer.
    auto undeallocate = [](Value buffer) {
      for (Operation *user : llvm::to_vector(buffer.getUsers()))
        if (isa<memref::DeallocOp>(user))
          user->erase();
    };

    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    for (func::FuncOp function : functions) {
      if (function.isExternal())
        continue;
      Block &body = function.getBody().front();

      // A transfer moves no data on the CPU: a thread reads and writes DDR
      // like the CPU, so the launches read and write the buffers of the
      // values themselves and the transfer leaves only its order, which the
      // launches and the batches keep. To a thread, the launches read the
      // source in place of the target; from a thread, the launch writes the
      // target in place of its own buffer (or the CPU reads that buffer in
      // place of the target, when the target is a buffer of its own). A
      // value passed between two threads: the job that reads the target
      // reads the source's buffer, and stream_transfer lets the core pass
      // it.
      DenseSet<std::pair<Value, int64_t>> passed;
      SmallVector<Value> unused;
      for (auto move : llvm::to_vector(body.getOps<TransferOp>())) {
        Value source = move.getSource(), target = move.getTarget();
        if (move.getTo()) {
          if (move.getFrom())
            passed.insert({source, *move.getTo()});
          undeallocate(target);
          target.replaceUsesWithIf(source, [&](OpOperand &use) {
            return isa<LaunchOp>(use.getOwner());
          });
          unused.push_back(target);
        } else if (target.getDefiningOp<memref::AllocOp>()) {
          undeallocate(target);
          target.replaceUsesWithIf(
              source, [&](OpOperand &use) { return use.getOwner() != move; });
          unused.push_back(target);
        } else {
          undeallocate(source);
          source.replaceUsesWithIf(
              target, [&](OpOperand &use) { return use.getOwner() != move; });
          unused.push_back(source);
        }
        move->erase();
      }
      for (Value buffer : unused) {
        if (!buffer.use_empty()) {
          buffer.getDefiningOp()->emitError("a buffer of a transfer in use");
          llvm::report_fatal_error("lower-core: a transfer buffer that "
                                   "another operation reads");
        }
        buffer.getDefiningOp<memref::AllocOp>().erase();
      }

      // Resolve transfer aliases before shortening CPU buffer lifetimes.
      OpPassManager lifetime(func::FuncOp::getOperationName());
      lifetime.addPass(bufferization::createOptimizeAllocationLivenessPass());
      if (failed(runPipeline(lifetime, function)))
        return signalPassFailure();

      // The open batch: its launches, the buffers each reads and writes,
      // its dependencies and transfers.
      SmallVector<LaunchOp> launches;
      SmallVector<SmallVector<Value>> reads, writes;
      DenseSet<Value> read, written, deferred;
      int64_t dependencies = 0, transfers = 0;
      // The call of each launch of the batch.
      SmallVector<Operation *> calls;
      SmallVector<Operation *> releases;
      auto close = [&](Operation *before) {
        OpBuilder builder(calls.front());
        Location loc = launches.front().getLoc();
        func::CallOp::create(builder, loc, begin,
                             ValueRange{constant(builder, loc, launches.size()),
                                        constant(builder, loc, dependencies),
                                        constant(builder, loc, transfers)});
        builder.setInsertionPointAfter(launches.back());
        func::CallOp::create(builder, loc, submit, ValueRange{});
        builder.setInsertionPoint(before);
        func::CallOp::create(builder, loc, wait, ValueRange{});
        for (Operation *release : releases)
          release->moveBefore(before);
        releases.clear();
        calls.clear();
        launches.clear();
        reads.clear();
        writes.clear();
        read.clear();
        written.clear();
        deferred.clear();
        dependencies = transfers = 0;
      };
      SmallVector<Operation *> ops;
      for (Operation &op : body)
        ops.push_back(&op);
      for (Operation *op : ops) {
        if (auto launch = dyn_cast<LaunchOp>(op)) {
          OpBuilder builder(launch);
          Location loc = launch.getLoc();
          auto callee = module.lookupSymbol<func::FuncOp>(launch.getCallee());
          bool needsHost =
              llvm::any_of(launch->getOperands(), [&](Value value) {
                return deferred.contains(root(value));
              });
          for (int64_t index :
               callee->getAttrOfType<DenseI64ArrayAttr>("core.host_inputs")
                   .asArrayRef())
            needsHost |= written.contains(root(launch->getOperand(index)));
          for (int64_t index :
               callee->getAttrOfType<DenseI64ArrayAttr>("core.host_writes")
                   .asArrayRef()) {
            Value buffer = root(launch->getOperand(index));
            needsHost |= written.contains(buffer) || read.contains(buffer);
          }
          if (!launches.empty() && needsHost)
            close(launch);
          SmallVector<Value> operands;
          for (auto [value, type] : llvm::zip_equal(launch->getOperands(),
                                                    callee.getArgumentTypes()))
            operands.push_back(
                rebase(builder, loc, value, cast<MemRefType>(type)));
          calls.push_back(func::CallOp::create(
              builder, loc,
              launcher(callee, signatures[launch.getThread()],
                       launch.getThread()),
              operands));
          SmallVector<Value> in, out;
          for (Value value : launch.getInputs())
            in.push_back(root(value));
          for (Value value : launch.getOuts())
            out.push_back(root(value));
          int64_t self = launches.size();
          // After the job: the values passed to it, the earlier jobs it
          // waits for.
          SmallVector<int64_t> sources;
          for (Value value : launch.getInputs()) {
            if (!passed.contains({value, launch.getThread()}))
              continue;
            int64_t source = 0;
            while (source < self &&
                   !llvm::is_contained(writes[source], root(value)))
              ++source;
            if (source == self) {
              launch->emitError("reads a value passed from a launch of an "
                                "earlier batch");
              llvm::report_fatal_error("lower-core: a value passed "
                                       "between threads outside a batch");
            }
            Value address = pointer(builder, loc, value);
            func::CallOp::create(builder, loc, pass,
                                 ValueRange{constant(builder, loc, source),
                                            constant(builder, loc, self),
                                            address, address,
                                            bytes(builder, loc, value)});
            sources.push_back(source);
            ++transfers;
          }
          auto touches = [](ArrayRef<Value> a, ArrayRef<Value> b) {
            return llvm::any_of(
                a, [&](Value value) { return llvm::is_contained(b, value); });
          };
          for (int64_t index = 0; index < self; ++index)
            if (!llvm::is_contained(sources, index) &&
                (touches(in, writes[index]) || touches(out, writes[index]) ||
                 touches(out, reads[index]))) {
              func::CallOp::create(builder, loc, after,
                                   ValueRange{constant(builder, loc, index)});
              ++dependencies;
            }
          launches.push_back(launch);
          reads.push_back(in);
          writes.push_back(out);
          read.insert(in.begin(), in.end());
          written.insert(out.begin(), out.end());
          if (callee->hasAttr("core.host_completion"))
            deferred.insert(out.begin(), out.end());
          continue;
        }
        if (launches.empty())
          continue;
        // The CPU waits for the batch before it touches a buffer a job
        // writes, or writes or frees one a job reads. Views of them touch
        // nothing.
        if (isa<ViewLikeOpInterface>(op))
          continue;
        // An asynchronous batch owns its buffers until the wait. Moving
        // their frees there also keeps independent launches in one batch.
        if (auto release = dyn_cast<memref::DeallocOp>(op)) {
          Value buffer = root(release.getMemref());
          if (read.contains(buffer) || written.contains(buffer)) {
            releases.push_back(op);
            continue;
          }
        }
        SetVector<Value> used(op->operand_begin(), op->operand_end());
        getUsedValuesDefinedAbove(op->getRegions(), used);
        bool touches = op->hasTrait<OpTrait::IsTerminator>();
        for (Value value : used) {
          if (!isa<MemRefType>(value.getType()))
            continue;
          Value buffer = root(value);
          if (written.contains(buffer))
            touches = true;
          if (read.contains(buffer) &&
              (!llvm::is_contained(op->getOperands(), value) ||
               mightHaveEffect<MemoryEffects::Write, MemoryEffects::Free>(
                   op, value)))
            touches = true;
        }
        if (touches)
          close(op);
      }

      for (Operation *op : ops)
        if (isa<LaunchOp>(op))
          op->erase();
    }
  }
};

PassRegistration<LowerCorePass> registration;
} // namespace

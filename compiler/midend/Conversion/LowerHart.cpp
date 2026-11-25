//===- LowerHart.cpp - The subgraphs of a graph on the kernel -------------===//
//
// lower-hart lowers the bufferized launches and transfers of a graph, placed
// by the design's convert-linalg-to-hart and after its deallocation, to the
// hart CPU and asynchronous jobs of the kernel
// (stack/kernel/include/stream.h):
//
//   transfer   dense buffers move no data on the CPU; strided views are
//              materialized as dense rows. A thread reads and writes DDR like
//              the CPU, so the launches read and write the buffers of the
//              values themselves, in their layout (rows that follow each
//              other, from the first element), and the transfer keeps only
//              its order. Between two threads, stream_transfer(source job,
//              target job, address, address, bytes) lets the hart pass the
//              value from one thread to the other (kernel_transfer)
//   launch     call <callee>_launch_<thread>(buffers): the pointer of every
//   buffer
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
// stream_job runs <callee>_thread on the hart CPU, which rebuilds the memrefs
// of the subgraph from the pointers and calls it; the Ball instructions the
// subgraph issues are the job's stream, which the thread executes once the
// batch is submitted. The signature is the thread's in design.pb, the same
// on every hart of the class (`hart`). A subgraph takes buffers of the
// identity layout: a buffer of another layout whose rows follow each other
// is passed as such a buffer from its first element, so a pointer and its
// dynamic sizes are the whole buffer.
//
//===----------------------------------------------------------------------===//

#include "Hart/HartOps.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/MemoryBuffer.h"

#include "design.pb.h"

using namespace mlir;
using namespace ::buddy::hart;

namespace mlir::buddy {
LogicalResult lowerThreadStream(func::FuncOp function);
}

namespace {
class LowerHartPass
    : public PassWrapper<LowerHartPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerHartPass)

  LowerHartPass() = default;
  LowerHartPass(const LowerHartPass &pass) : PassWrapper(pass) {}

  StringRef getArgument() const final { return "lower-hart"; }
  StringRef getDescription() const final {
    return "Lower the launches and transfers of a graph to DDR buffers and "
           "the kernel of its hart";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect,
                    memref::MemRefDialect, linalg::LinalgDialect>();
  }

  Option<std::string> designPb{*this, "design",
                               llvm::cl::desc("design.pb of the device")};
  Option<std::string> hartClass{
      *this, "hart", llvm::cl::desc("class of the hart the graph runs on")};

  void runOnOperation() override {
    auto buffer = llvm::MemoryBuffer::getFile(designPb);
    if (!buffer)
      llvm::report_fatal_error(llvm::Twine("cannot read ") + designPb);
    buckyball::config::Design design;
    if (!design.ParseFromArray((*buffer)->getBufferStart(),
                               (*buffer)->getBufferSize()))
      llvm::report_fatal_error(llvm::Twine("cannot parse ") + designPb);
    // The signature of each thread of the hart class, by its index in the
    // hart.
    SmallVector<uint64_t> signatures;
    SmallVector<buckyball::config::DataLink> links;
    int64_t bankBytes = 0;
    for (const auto &hart : design.device().harts()) {
      if (hart.component() != hartClass)
        continue;
      SmallVector<uint64_t> threads;
      for (uint32_t index : hart.thread_indices())
        threads.push_back(design.threads(index).signature());
      if (signatures.empty()) {
        for (const auto &link : hart.data_links())
          links.push_back(link);
        for (uint32_t index : hart.thread_indices()) {
          const auto &bank = design.threads(index).mem().bank();
          int64_t bytes = int64_t(bank.width()) * bank.entries() / 8;
          if (bankBytes && bankBytes != bytes)
            llvm::report_fatal_error("thread data banks have different sizes");
          bankBytes = bytes;
        }
      }
      if (!signatures.empty() && signatures != threads)
        llvm::report_fatal_error(llvm::Twine("the harts of class ") +
                                 hartClass + " differ in their threads");
      signatures = threads;
    }
    if (signatures.empty())
      llvm::report_fatal_error(llvm::Twine("no hart of class ") + hartClass +
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
    func::FuncOp job = declare("stream_job", {i64, i32, entryType, ptr, i32});
    func::FuncOp after = declare("stream_after", {i64});
    // stream_transfer only where values pass between two threads.
    func::FuncOp pass = declare("stream_transfer", {i64, i64, ptr, ptr, i64});
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

    // Sliced subgraphs can leave unused operands and destinations. Drop them
    // before constructing the opaque kernel frame, where LLVM cannot see
    // which descriptor slots the thread entry reads.
    for (Operation *operation : recorded) {
      auto callee = cast<func::FuncOp>(operation);
      if (callee->hasAttr("hart.native_source"))
        continue;
      llvm::BitVector unused(callee.getNumArguments());
      SmallVector<int64_t> remap(callee.getNumArguments(), -1);
      int64_t next = 0;
      for (auto argument : callee.getArguments()) {
        unsigned index = argument.getArgNumber();
        unused[index] = argument.use_empty();
        if (!unused[index])
          remap[index] = next++;
      }
      if (unused.none())
        continue;
      module.walk([&](LaunchOp launch) {
        if (launch.getCallee() != callee.getName())
          return;
        SmallVector<Value> inputs, outs;
        unsigned inputCount = launch.getInputs().size();
        for (auto [index, input] : llvm::enumerate(launch.getInputs()))
          if (!unused[index])
            inputs.push_back(input);
        for (auto [index, out] : llvm::enumerate(launch.getOuts()))
          if (!unused[inputCount + index])
            outs.push_back(out);
        launch.getInputsMutable().assign(inputs);
        launch.getOutsMutable().assign(outs);
      });
      for (StringRef name : {"hart.host_inputs", "hart.host_writes"}) {
        SmallVector<int64_t> indices;
        for (int64_t index :
             callee->getAttrOfType<DenseI64ArrayAttr>(name).asArrayRef())
          if (!unused[index])
            indices.push_back(remap[index]);
        callee->setAttr(name, DenseI64ArrayAttr::get(context, indices));
      }
      callee.eraseArguments(unused);
    }

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
          arith::ConstantIndexOp::create(
              builder, loc, (type.getElementTypeBitWidth() + 7) / 8));
      Value word = arith::IndexCastOp::create(
          builder, loc, i64, arith::AddIOp::create(builder, loc, base, offset));
      return LLVM::IntToPtrOp::create(builder, loc, ptr, word).getResult();
    };
    // The bytes of a buffer of the identity layout.
    auto bytes = [&](OpBuilder &builder, Location loc, Value buffer) {
      auto type = cast<MemRefType>(buffer.getType());
      Value count = arith::ConstantIndexOp::create(
          builder, loc, (type.getElementTypeBitWidth() + 7) / 8);
      for (int64_t axis = 0; axis < type.getRank(); ++axis)
        count = arith::MulIOp::create(
            builder, loc, count,
            memref::DimOp::create(builder, loc, buffer, axis));
      return arith::IndexCastOp::create(builder, loc, i64, count).getResult();
    };

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
        Value allocated;
        if (callee->hasAttr("hart.native_source"))
          allocated = word(next++, ptr);
        Value pointer = word(next++, ptr);
        auto [layoutStrides, layoutOffset] = memref.getStridesAndOffset();
        Value offset = ShapedType::isDynamic(layoutOffset)
                           ? word(next++, i64)
                           : Value(LLVM::ConstantOp::create(
                                 builder, loc, i64,
                                 builder.getI64IntegerAttr(layoutOffset)));
        Value descriptor;
        if (memref.hasStaticShape() && memref.getLayout().isIdentity()) {
          descriptor = MemRefDescriptor::fromStaticShape(
              builder, loc, converter, memref, pointer);
        } else {
          // Row-major strides over the sizes, the dynamic ones from the
          // words.
          MemRefDescriptor built = MemRefDescriptor::poison(
              builder, loc, converter.convertType(memref));
          built.setAllocatedPtr(builder, loc, pointer);
          built.setAlignedPtr(builder, loc, pointer);
          built.setOffset(builder, loc, offset);
          SmallVector<Value> sizes;
          for (int64_t size : memref.getShape())
            sizes.push_back(
                ShapedType::isDynamic(size)
                    ? word(next++, i64)
                    : LLVM::ConstantOp::create(
                          builder, loc, i64, builder.getI64IntegerAttr(size)));
          SmallVector<Value> strides(memref.getRank());
          if (memref.getLayout().isIdentity()) {
            Value stride = LLVM::ConstantOp::create(
                builder, loc, i64, builder.getI64IntegerAttr(1));
            for (int64_t axis = memref.getRank() - 1; axis >= 0; --axis) {
              strides[axis] = stride;
              stride = LLVM::MulOp::create(builder, loc, stride, sizes[axis]);
            }
          } else {
            for (auto [axis, stride] :
                 llvm::enumerate(memref.getStridesAndOffset().first))
              strides[axis] = ShapedType::isDynamic(stride)
                                  ? word(next++, i64)
                                  : LLVM::ConstantOp::create(
                                        builder, loc, i64,
                                        builder.getI64IntegerAttr(stride));
          }
          for (int64_t axis = 0; axis < memref.getRank(); ++axis) {
            built.setSize(builder, loc, axis, sizes[axis]);
            built.setStride(builder, loc, axis, strides[axis]);
          }
          descriptor = built;
        }
        if (allocated) {
          MemRefDescriptor built(descriptor);
          built.setAllocatedPtr(builder, loc, allocated);
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

    // <callee>_launch_<thread>(buffers): their pointers, then stream_job on
    // that thread. A subgraph launched on several threads (the row parts of
    // a product) has a launcher for each.
    auto launcher = [&](func::FuncOp callee, uint64_t signature, int64_t thread,
                        unsigned inputs) {
      std::string name =
          (callee.getName() + "_launch_" + llvm::Twine(thread)).str();
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
      if (callee->hasAttr("hart.native_source"))
        words += callee.getNumArguments();
      for (Type type : callee.getArgumentTypes()) {
        auto memory = cast<MemRefType>(type);
        words += memory.getNumDynamicDims();
        if (!memory.getLayout().isIdentity()) {
          auto [strides, offset] = memory.getStridesAndOffset();
          words += ShapedType::isDynamic(offset);
          for (int64_t stride : strides)
            words += ShapedType::isDynamic(stride);
        }
      }
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
      // Only zero-element outputs prove a compute job empty. A job without
      // outputs can still perform control work.
      Value empty = arith::ConstantIntOp::create(
          builder, loc, function.getNumArguments() > inputs, 1);
      for (auto [argument, buffer] : llvm::enumerate(function.getArguments())) {
        auto type = cast<MemRefType>(buffer.getType());
        if (callee->hasAttr("hart.native_source")) {
          Value descriptor =
              UnrealizedConversionCastOp::create(
                  builder, loc, converter.convertType(type), buffer)
                  .getResult(0);
          store(next++,
                MemRefDescriptor(descriptor).allocatedPtr(builder, loc));
        }
        auto metadata =
            memref::ExtractStridedMetadataOp::create(builder, loc, buffer);
        if (argument >= inputs) {
          Value elements = arith::ConstantIndexOp::create(builder, loc, 1);
          for (Value size : metadata.getSizes())
            elements = arith::MulIOp::create(builder, loc, elements, size);
          Value zero = arith::CmpIOp::create(
              builder, loc, arith::CmpIPredicate::eq, elements,
              arith::ConstantIndexOp::create(builder, loc, 0));
          empty = arith::AndIOp::create(builder, loc, empty, zero);
        }
        Value base = memref::ExtractAlignedPointerAsIndexOp::create(
            builder, loc, buffer);
        store(next++, LLVM::IntToPtrOp::create(
                          builder, loc, ptr,
                          arith::IndexCastOp::create(builder, loc, i64, base)
                              .getResult()));
        if (ShapedType::isDynamic(type.getStridesAndOffset().second))
          store(next++, arith::IndexCastOp::create(builder, loc, i64,
                                                   metadata.getOffset()));
        for (int64_t axis = 0; axis < type.getRank(); ++axis)
          if (type.isDynamicDim(axis))
            store(next++,
                  arith::IndexCastOp::create(
                      builder, loc, i64,
                      memref::DimOp::create(builder, loc, buffer, axis)));
        if (!type.getLayout().isIdentity()) {
          for (auto [axis, stride] :
               llvm::enumerate(type.getStridesAndOffset().first))
            if (ShapedType::isDynamic(stride))
              store(next++,
                    arith::IndexCastOp::create(builder, loc, i64,
                                               metadata.getStrides()[axis]));
        }
      }
      Value body = func::ConstantOp::create(builder, loc, entryType,
                                            entry(callee).getName());
      Value code = arith::ConstantIntOp::create(
          builder, loc, static_cast<int64_t>(signature), 64);
      Value index = arith::ConstantIntOp::create(builder, loc, thread, 32);
      func::CallOp::create(
          builder, loc, job,
          ValueRange{code, index, body, pointers,
                     arith::ExtUIOp::create(builder, loc, i32, empty)});
      func::ReturnOp::create(builder, loc);
      return function;
    };

    module.walk([&](Operation *op) {
      if (isa<LaunchOp, TransferOp>(op) &&
          !isa<func::FuncOp>(op->getParentOp())) {
        op->emitError("a launch or transfer inside a region");
        llvm::report_fatal_error("lower-hart: launches and transfers "
                                 "stand in the body of a function");
      }
    });

    auto contiguous = [](MemRefType type) {
      auto [strides, offset] = type.getStridesAndOffset();
      int64_t expected = 1;
      for (int64_t axis = type.getRank() - 1; axis >= 0; --axis) {
        int64_t size = type.getDimSize(axis);
        if (size != 1 && strides[axis] != expected)
          return false;
        expected =
            ShapedType::isDynamic(size) || ShapedType::isDynamic(expected)
                ? ShapedType::kDynamic
                : expected * size;
      }
      return true;
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
      if (!type.getLayout().isIdentity())
        return memref::CastOp::create(builder, loc, type, buffer);
      if (!contiguous(from)) {
        mlir::emitError(loc, "a thread reads ") << from;
        llvm::report_fatal_error("lower-hart: a thread reads a buffer "
                                 "whose rows do not follow each other");
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
    auto overlaps = [&](Value left, Value right) {
      if (root(left) != root(right))
        return false;
      SmallVector<std::pair<int64_t, int64_t>> spans;
      for (Value value : {left, right}) {
        for (Value view = value;
             auto op = view.getDefiningOp<ViewLikeOpInterface>();
             view = op.getViewSource())
          if (isa<memref::ViewOp>(op.getOperation()))
            return true;
        auto type = cast<MemRefType>(value.getType());
        SmallVector<int64_t> strides;
        int64_t offset;
        if (!type.hasStaticShape() ||
            failed(type.getStridesAndOffset(strides, offset)) ||
            ShapedType::isDynamic(offset) || !contiguous(type))
          return true;
        int64_t bytes = (type.getElementTypeBitWidth() + 7) / 8;
        spans.push_back(
            {offset * bytes, (offset + type.getNumElements()) * bytes});
      }
      return spans[0].first < spans[1].second &&
             spans[1].first < spans[0].second;
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

      // Bufferization can leave a value assembled in a temporary allocation
      // and copied whole, through a view, into a result of the function: the
      // allocation is the result itself, viewed as the allocation.
      for (auto copy : llvm::to_vector(body.getOps<memref::CopyOp>())) {
        auto target = dyn_cast<BlockArgument>(copy.getTarget());
        auto view = copy.getSource().getDefiningOp<ViewLikeOpInterface>();
        auto allocation =
            view ? view.getViewSource().getDefiningOp<memref::AllocOp>()
                 : nullptr;
        if (!target || target.getOwner() != &body || !allocation)
          continue;
        auto targetType = cast<MemRefType>(target.getType());
        auto allocationType = allocation.getType();
        auto sourceType = cast<MemRefType>(copy.getSource().getType());
        if (!targetType.getLayout().isIdentity() ||
            !allocationType.getLayout().isIdentity() ||
            !contiguous(sourceType) || !targetType.hasStaticShape() ||
            sourceType.getStridesAndOffset().second != 0 ||
            targetType.getNumElements() != allocationType.getNumElements() ||
            !llvm::all_of(allocation->getUsers(),
                          [&](Operation *user) {
                            return isa<ViewLikeOpInterface, memref::DeallocOp>(
                                user);
                          }) ||
            !copy.getSource().hasOneUse())
          continue;
        OpBuilder builder(allocation);
        SmallVector<int64_t> strides(allocationType.getRank());
        int64_t stride = 1;
        for (int64_t axis = allocationType.getRank() - 1; axis >= 0; --axis) {
          strides[axis] = stride;
          stride *= allocationType.getDimSize(axis);
        }
        Value place = memref::ReinterpretCastOp::create(
            builder, allocation.getLoc(), allocationType, target, 0,
            allocationType.getShape(), strides);
        for (Operation *user : llvm::to_vector(allocation->getUsers()))
          if (isa<memref::DeallocOp>(user))
            user->erase();
        Operation *source = copy.getSource().getDefiningOp();
        copy->erase();
        source->erase();
        allocation.getResult().replaceAllUsesWith(place);
        allocation->erase();
      }

      // Bufferization can leave a destination slice in a temporary allocation.
      // Forward its sole transfer directly into the final slice before
      // aliasing.
      for (auto copy : llvm::to_vector(body.getOps<memref::CopyOp>())) {
        Value source = copy.getSource();
        auto view = source.getDefiningOp<ViewLikeOpInterface>();
        if (!view || !view.getViewSource().getDefiningOp<memref::AllocOp>())
          continue;
        if (!llvm::all_of(view.getViewSource().getUsers(),
                          [&](Operation *user) {
                            return user == view.getOperation() ||
                                   isa<memref::DeallocOp>(user);
                          }))
          continue;
        TransferOp writer;
        bool exclusive = true;
        for (Operation *user : source.getUsers()) {
          if (user == copy)
            continue;
          auto transfer = dyn_cast<TransferOp>(user);
          if (!transfer || !transfer.getFrom() ||
              transfer.getTarget() != source || writer) {
            exclusive = false;
            break;
          }
          writer = transfer;
        }
        if (exclusive && writer) {
          writer.getTargetMutable().assign(copy.getTarget());
          copy.erase();
        }
      }

      // Contiguous transfers move no data on the CPU: a thread reads and writes
      // DDR like the CPU, so the launches read and write the buffers of the
      // values themselves and the transfer leaves only its order, which the
      // launches and the batches keep. To a thread, the launches read the
      // source in place of the target; from a thread, the launch writes the
      // target in place of its own buffer (or the CPU reads that buffer in
      // place of the target, when the target is a buffer of its own). A
      // value passed between two threads: the job that reads the target
      // reads the source's buffer, and stream_transfer lets the hart pass
      // it.
      DenseSet<Value> returned;
      for (Value value : body.getTerminator()->getOperands())
        if (isa<MemRefType>(value.getType()))
          returned.insert(root(value));
      SmallVector<Value> unused;
      // The allocation and pure view expressions of a transfer destination,
      // moved before `before` (a destination slice can be formed after the
      // launch that writes it, or after the transfer that fills it).
      std::function<void(Value, Operation *)> hoist = [&](Value value,
                                                          Operation *before) {
        Operation *def = value.getDefiningOp();
        if (!def || def->getBlock() != &body || def->isBeforeInBlock(before))
          return;
        if (!isa<memref::AllocOp>(def) && !isPure(def))
          llvm::report_fatal_error(
              "lower-hart: transfer destination depends on CPU execution");
        for (Value operand : def->getOperands())
          hoist(operand, before);
        def->moveBefore(before);
      };
      for (auto move : llvm::to_vector(body.getOps<TransferOp>())) {
        Value source = move.getSource(), target = move.getTarget();
        // A strided read can alias DDR when every consuming entry preserves its
        // strides.
        auto sourceType = cast<MemRefType>(source.getType());
        bool stridedRead = move.getTo() && !move.getFrom() &&
                           sourceType.getRank() > 0 &&
                           sourceType.getStridesAndOffset().first.back() == 1;
        for (OpOperand &use : target.getUses())
          if (auto launch = dyn_cast<LaunchOp>(use.getOwner())) {
            auto callee = module.lookupSymbol<func::FuncOp>(launch.getCallee());
            stridedRead &=
                !cast<MemRefType>(
                     callee.getArgumentTypes()[use.getOperandNumber()])
                     .getLayout()
                     .isIdentity();
          }
        auto targetType = cast<MemRefType>(target.getType());
        bool stridedWrite = move.getFrom() && !move.getTo() &&
                            targetType.getRank() > 0 &&
                            targetType.getStridesAndOffset().first.back() == 1;
        bool producerWrites = false;
        for (OpOperand &use : source.getUses())
          if (auto launch = dyn_cast<LaunchOp>(use.getOwner())) {
            producerWrites |=
                cast<DestinationStyleOpInterface>(launch.getOperation())
                    .isDpsInit(&use);
            auto callee = module.lookupSymbol<func::FuncOp>(launch.getCallee());
            stridedWrite &=
                !cast<MemRefType>(
                     callee.getArgumentTypes()[use.getOperandNumber()])
                     .getLayout()
                     .isIdentity();
          }
        stridedWrite &= producerWrites;
        if (!stridedRead && !stridedWrite &&
            (!contiguous(cast<MemRefType>(source.getType())) ||
             !contiguous(cast<MemRefType>(target.getType())))) {
          hoist(target, move);
          OpBuilder builder(move);
          linalg::CopyOp::create(builder, move.getLoc(), source, target);
          move->erase();
          continue;
        }
        if (move.getTo()) {
          undeallocate(target);
          target.replaceUsesWithIf(source, [&](OpOperand &use) {
            return isa<LaunchOp>(use.getOwner());
          });
          unused.push_back(target);
        } else if (target.getDefiningOp<memref::AllocOp>()) {
          if (returned.contains(root(target))) {
            undeallocate(source);
            returned.insert(root(source));
          }
          undeallocate(target);
          target.replaceUsesWithIf(
              source, [&](OpOperand &use) { return use.getOwner() != move; });
          unused.push_back(target);
        } else {
          // A destination slice can be formed after its producer launch.
          // Make its allocation and pure view expressions dominate that launch
          // before redirecting the producer's output into the slice.
          Operation *before = move;
          for (Operation *user : source.getUsers())
            if (user->getBlock() == &body && user->isBeforeInBlock(before))
              before = user;
          hoist(target, before);
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
          llvm::report_fatal_error("lower-hart: a transfer buffer that "
                                   "another operation reads");
        }
        buffer.getDefiningOp<memref::AllocOp>().erase();
      }

      // Resolve transfer aliases before shortening CPU buffer lifetimes.
      OpPassManager lifetime(func::FuncOp::getOperationName());
      // Forwarding can leave copies between identical views. Eliminate them
      // before scheduling, so they cannot split independent launches into
      // batches.
      lifetime.addPass(createCSEPass());
      lifetime.addPass(createCanonicalizerPass());
      lifetime.addPass(createCSEPass());
      lifetime.addPass(createCanonicalizerPass());
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
      SmallVector<SmallVector<std::pair<int64_t, func::CallOp>>> afters;
      auto close = [&](Operation *before) {
        OpBuilder builder(launches.back());
        Location loc = launches.front().getLoc();
        builder.setInsertionPointAfter(launches.back());
        auto submitted =
            func::CallOp::create(builder, loc, submit, ValueRange{});
        builder.setInsertionPoint(submitted);
        // Logical links expose a shared bank budget, not its storage or
        // routing. Only accelerator-only values wholly described by this batch
        // can live on a link. CPU-visible values retain their ordinary memory
        // contract.
        DenseSet<Value> host;
        for (Operation &operation : body) {
          if (llvm::is_contained(calls, &operation))
            continue;
          if (auto launch = dyn_cast<LaunchOp>(operation)) {
            auto callee = module.lookupSymbol<func::FuncOp>(launch.getCallee());
            for (StringRef name : {"hart.host_inputs", "hart.host_writes"})
              for (int64_t index :
                   callee->getAttrOfType<DenseI64ArrayAttr>(name).asArrayRef())
                host.insert(root(launch->getOperand(index)));
            continue;
          }
          if (isa<ViewLikeOpInterface, memref::DeallocOp>(operation) ||
              isMemoryEffectFree(&operation))
            continue;
          SetVector<Value> used(operation.operand_begin(),
                                operation.operand_end());
          getUsedValuesDefinedAbove(operation.getRegions(), used);
          for (Value value : used)
            if (isa<MemRefType>(value.getType()))
              host.insert(root(value));
        }
        struct Region {
          Value value, base;
          OpFoldResult offset;
          SmallVector<OpFoldResult> sizes;
        };
        SmallVector<Region> regions;
        auto range = [&](Value value) {
          while (auto cast = value.getDefiningOp<memref::CastOp>())
            value = cast.getSource();
          auto type = cast<MemRefType>(value.getType());
          if (!contiguous(type))
            return value;
          Region region{value, root(value), builder.getIndexAttr(0), {}};
          if (auto view = value.getDefiningOp<memref::ReinterpretCastOp>()) {
            region.offset = view.getMixedOffsets().front();
            region.sizes = view.getMixedSizes();
          } else if (auto alloc = value.getDefiningOp<memref::AllocOp>()) {
            size_t dynamic = 0;
            for (int64_t size : type.getShape())
              region.sizes.push_back(
                  ShapedType::isDynamic(size)
                      ? OpFoldResult(alloc.getDynamicSizes()[dynamic++])
                      : OpFoldResult(builder.getIndexAttr(size)));
          } else {
            return value;
          }
          if (auto offset = getConstantIntValue(region.offset))
            region.offset = builder.getIndexAttr(*offset);
          for (auto &size : region.sizes)
            if (auto constant = getConstantIntValue(size))
              size = builder.getIndexAttr(*constant);
          llvm::erase_if(region.sizes, [](OpFoldResult size) {
            return getConstantIntValue(size) == 1;
          });
          for (const auto &known : regions)
            if (known.base == region.base && known.offset == region.offset &&
                known.sizes == region.sizes &&
                cast<MemRefType>(known.value.getType()).getElementType() ==
                    type.getElementType())
              return known.value;
          regions.push_back(region);
          return value;
        };
        SetVector<Value> roots;
        for (const auto &outputs : writes)
          for (Value value : outputs)
            roots.insert(root(value));
        SmallVector<Value> occupied;
        SmallVector<int64_t> entries(links.size(), 0);
        for (size_t link = 0; link < links.size(); ++link)
          occupied.push_back(constant(builder, loc, 0));
        for (Value buffer : roots) {
          auto allocation = buffer.getDefiningOp<memref::AllocOp>();
          if (!allocation || host.contains(buffer) || returned.contains(buffer))
            continue;
          struct Edge {
            int64_t source, target;
            Value value;
            int64_t offset = 0, bytes = -1;
          };
          SmallVector<Edge> edges;
          bool valid = true;
          for (auto launch : body.getOps<LaunchOp>())
            if (!llvm::is_contained(launches, launch))
              for (Value value : launch->getOperands())
                valid &= root(value) != buffer;
          // Static contiguous views share one partition of their byte ranges.
          // A producer or consumer can span several adjacent edges without CPU
          // copies.
          DenseMap<Value, std::pair<int64_t, int64_t>> spans;
          SmallVector<int64_t> boundaries;
          bool staticRanges = true;
          for (const auto *operands : {&reads, &writes})
            for (const auto &values : *operands)
              for (Value value : values) {
                if (root(value) != buffer || spans.contains(value))
                  continue;
                Value canonical = range(value);
                auto found = llvm::find_if(regions, [&](const Region &region) {
                  return region.value == canonical;
                });
                if (found == regions.end()) {
                  staticRanges = false;
                  continue;
                }
                auto offset = getConstantIntValue(found->offset);
                int64_t length = (cast<MemRefType>(value.getType())
                                      .getElementTypeBitWidth() +
                                  7) /
                                 8;
                int64_t elementBytes = length;
                for (OpFoldResult size : found->sizes) {
                  auto dimension = getConstantIntValue(size);
                  if (!dimension) {
                    staticRanges = false;
                    break;
                  }
                  length *= *dimension;
                }
                if (!offset) {
                  staticRanges = false;
                  continue;
                }
                int64_t start = *offset * elementBytes;
                spans[value] = {start, start + length};
                boundaries.push_back(start);
                boundaries.push_back(start + length);
              }
          if (staticRanges) {
            llvm::sort(boundaries);
            boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                             boundaries.end());
            for (int64_t target = 0; target < int64_t(launches.size());
                 ++target) {
              DenseSet<int64_t> consumed;
              for (Value input : reads[target]) {
                if (root(input) != buffer)
                  continue;
                auto [begin, end] = spans.lookup(input);
                for (size_t part = 0; part + 1 < boundaries.size(); ++part) {
                  int64_t first = boundaries[part], last = boundaries[part + 1];
                  if (first < begin || last > end || first == last ||
                      !consumed.insert(first).second)
                    continue;
                  int64_t producer = -1;
                  for (int64_t source = 0; source < target; ++source) {
                    bool covers =
                        llvm::any_of(writes[source], [&](Value output) {
                          if (root(output) != buffer)
                            return false;
                          auto span = spans.lookup(output);
                          return span.first <= first && last <= span.second;
                        });
                    if (covers) {
                      if (producer >= 0)
                        valid = false;
                      producer = source;
                    }
                  }
                  if (producer < 0 || first % 16 || (last - first) % 16)
                    valid = false;
                  else
                    edges.push_back(
                        {producer, target, buffer, first, last - first});
                }
              }
            }
          } else {
            for (int64_t target = 0; target < int64_t(launches.size());
                 ++target) {
              DenseSet<Value> seen;
              for (Value input : reads[target]) {
                if (root(input) != buffer || !seen.insert(range(input)).second)
                  continue;
                int64_t producer = -1;
                for (int64_t source = 0; source < target; ++source)
                  for (Value output : writes[source])
                    if (range(output) == range(input)) {
                      if (producer >= 0)
                        valid = false;
                      producer = source;
                    }
                if (producer < 0)
                  valid = false;
                else
                  edges.push_back({producer, target, input});
              }
            }
          }
          for (int64_t a = 0; a < int64_t(launches.size()); ++a)
            for (int64_t z = a + 1; z < int64_t(launches.size()); ++z)
              for (Value left : writes[a])
                for (Value right : writes[z]) {
                  if (root(left) != buffer || root(right) != buffer ||
                      !overlaps(left, right))
                    continue;
                  auto first = module.lookupSymbol<func::FuncOp>(
                      launches[a].getCallee());
                  auto last = module.lookupSymbol<func::FuncOp>(
                      launches[z].getCallee());
                  auto group = first->getAttr("hart.parallel_group");
                  valid &= group &&
                           group == last->getAttr("hart.parallel_group") &&
                           first->getAttr("hart.parallel_part") !=
                               last->getAttr("hart.parallel_part");
                }
          if (!valid || edges.empty())
            continue;
          int64_t link = 0;
          for (; link < int64_t(links.size()); ++link) {
            bool connected = true;
            for (const auto &edge : edges)
              connected &=
                  llvm::is_contained(links[link].threads(),
                                     launches[edge.source].getThread()) &&
                  llvm::is_contained(links[link].threads(),
                                     launches[edge.target].getThread());
            if (connected &&
                entries[link] + int64_t(edges.size()) <= links[link].banks())
              break;
          }
          if (link == int64_t(links.size()))
            continue;
          SmallVector<Value> addresses, sizes;
          SmallVector<size_t> payloads;
          Value needed = constant(builder, loc, 0);
          Value aligned = arith::ConstantIntOp::create(builder, loc, 1, 1);
          for (auto [index, edge] : llvm::enumerate(edges)) {
            auto known = llvm::find_if(payloads, [&](size_t previous) {
              const auto &other = edges[previous];
              return staticRanges ? edge.offset == other.offset &&
                                        edge.bytes == other.bytes
                                  : range(edge.value) == range(other.value);
            });
            if (known != payloads.end()) {
              addresses.push_back(addresses[*known]);
              sizes.push_back(sizes[*known]);
              continue;
            }
            payloads.push_back(index);
            Value value = range(edge.value);
            valid &= contiguous(cast<MemRefType>(value.getType()));
            Value size = staticRanges ? constant(builder, loc, edge.bytes)
                                      : bytes(builder, loc, value);
            Value address = pointer(builder, loc, value);
            if (staticRanges && edge.offset)
              address = LLVM::GEPOp::create(
                  builder, loc, ptr, builder.getI8Type(), address,
                  ValueRange{constant(builder, loc, edge.offset)});
            addresses.push_back(address);
            sizes.push_back(size);
            Value banks = arith::DivUIOp::create(
                builder, loc,
                arith::AddIOp::create(builder, loc, size,
                                      constant(builder, loc, bankBytes - 1)),
                constant(builder, loc, bankBytes));
            needed = arith::AddIOp::create(builder, loc, needed, banks);
            Value integer =
                LLVM::PtrToIntOp::create(builder, loc, i64, address);
            Value low = arith::AndIOp::create(
                builder, loc, arith::OrIOp::create(builder, loc, integer, size),
                constant(builder, loc, 15));
            aligned = arith::AndIOp::create(
                builder, loc, aligned,
                arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                      low, constant(builder, loc, 0)));
          }
          if (!valid)
            continue;
          Value total =
              arith::AddIOp::create(builder, loc, occupied[link], needed);
          Value enabled = arith::AndIOp::create(
              builder, loc, aligned,
              arith::CmpIOp::create(
                  builder, loc, arith::CmpIPredicate::ule, total,
                  constant(builder, loc, links[link].banks())));
          occupied[link] = arith::SelectOp::create(builder, loc, enabled, total,
                                                   occupied[link]);
          for (auto [index, edge] : llvm::enumerate(edges)) {
            Value address = addresses[index], size = sizes[index];
            func::CallOp::create(
                builder, loc, pass,
                ValueRange{constant(builder, loc, edge.source),
                           constant(builder, loc, edge.target), address,
                           address,
                           arith::SelectOp::create(builder, loc, enabled, size,
                                                   constant(builder, loc, 0))});
            for (auto &[source, call] : afters[edge.target])
              if (source == edge.source && call) {
                call.erase();
                call = {};
                --dependencies;
              }
            ++transfers;
          }
          entries[link] += edges.size();
        }
        builder.setInsertionPoint(calls.front());
        func::CallOp::create(builder, loc, begin,
                             ValueRange{constant(builder, loc, launches.size()),
                                        constant(builder, loc, dependencies),
                                        constant(builder, loc, transfers)});
        builder.setInsertionPoint(before);
        func::CallOp::create(builder, loc, wait, ValueRange{});
        for (Operation *release : releases)
          release->moveBefore(before);
        releases.clear();
        afters.clear();
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
                return llvm::any_of(deferred, [&](Value pending) {
                  return overlaps(value, pending);
                });
              });
          for (int64_t index :
               callee->getAttrOfType<DenseI64ArrayAttr>("hart.host_inputs")
                   .asArrayRef())
            needsHost |= written.contains(root(launch->getOperand(index)));
          for (int64_t index :
               callee->getAttrOfType<DenseI64ArrayAttr>("hart.host_writes")
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
                       launch.getThread(), launch.getInputs().size()),
              operands));
          SmallVector<Value> in, out;
          for (Value value : launch.getInputs())
            in.push_back(value);
          for (Value value : launch.getOuts())
            out.push_back(value);
          int64_t self = launches.size();
          auto touches = [&](ArrayRef<Value> a, ArrayRef<Value> b) {
            return llvm::any_of(a, [&](Value value) {
              return llvm::any_of(
                  b, [&](Value other) { return overlaps(value, other); });
            });
          };
          SmallVector<std::pair<int64_t, func::CallOp>> waits;
          for (int64_t index = 0; index < self; ++index) {
            auto earlier =
                module.lookupSymbol<func::FuncOp>(launches[index].getCallee());
            auto group = callee->getAttr("hart.parallel_group");
            if (group && group == earlier->getAttr("hart.parallel_group") &&
                callee->getAttr("hart.parallel_part") !=
                    earlier->getAttr("hart.parallel_part"))
              continue;
            if (touches(in, writes[index]) || touches(out, writes[index]) ||
                touches(out, reads[index])) {
              waits.push_back(
                  {index, func::CallOp::create(
                              builder, loc, after,
                              ValueRange{constant(builder, loc, index)})});
              ++dependencies;
            }
          }
          launches.push_back(launch);
          afters.push_back(waits);
          reads.push_back(in);
          writes.push_back(out);
          for (Value value : in)
            read.insert(root(value));
          for (Value value : out)
            written.insert(root(value));
          if (callee->hasAttr("hart.host_completion"))
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
      // Transfer forwarding can make two views identical. Fold the resulting
      // self-copies before they become opaque runtime calls or scalar loops.
      OpPassManager cleanup(func::FuncOp::getOperationName());
      cleanup.addPass(createCanonicalizerPass());
      cleanup.addPass(createCSEPass());
      cleanup.addPass(createCanonicalizerPass());
      if (failed(runPipeline(cleanup, function)))
        return signalPassFailure();
    }
    // Scheduling consumed these annotations. Equivalent recorder bodies can
    // now share CPU code even when their jobs belong to different threads.
    for (Operation *function : recorded)
      for (StringRef name : {"bbacc.target", "hart.program_bank",
                             "hart.parallel_group", "hart.parallel_part"})
        function->removeAttr(name);
    // The generic deduplication pass also visits public functions. Preserve
    // their exported identities while merging private recorder bodies.
    SmallVector<func::FuncOp> exported;
    for (auto function : module.getOps<func::FuncOp>())
      if (!function.isPrivate()) {
        function->setAttr("buckyball.public_symbol", function.getSymNameAttr());
        exported.push_back(function);
      }
    OpPassManager sharedCode(ModuleOp::getOperationName());
    sharedCode.addPass(func::createDuplicateFunctionEliminationPass());
    if (failed(runPipeline(sharedCode, module)))
      return signalPassFailure();
    for (auto function : exported)
      function->removeAttr("buckyball.public_symbol");
  }
};

PassRegistration<LowerHartPass> registration;
} // namespace

namespace mlir::buddy {
std::unique_ptr<Pass> createLowerHartPass(StringRef design, StringRef hart) {
  auto pass = std::make_unique<LowerHartPass>();
  pass->designPb = design.str();
  pass->hartClass = hart.str();
  return pass;
}
} // namespace mlir::buddy

//===- LowerHartNative.cpp - Compile subgraphs to Ant programs -----------===//
// The existing hart lowering owns batching, dependencies and logical
// transfers. This experimental backend replaces its stream recording with
// typed frames and ordinary kernel jobs. Preparation and completion remain
// on the host; Ant scalar memory accesses are confined to its frame/stack.

#include "Hart/HartOps.h"
#include "design.pb.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace ::buddy::hart;

namespace mlir::buddy {
std::unique_ptr<Pass> createLowerHartPass(StringRef design, StringRef hart);
}

namespace {
struct NativeBody {
  func::FuncOp source, prepare, complete;
  llvm::SetVector<Value> captures;
  DenseSet<Operation *> ant, completion;
  SmallVector<Type> fields;
  int64_t inputWords = 0, words = 0;
  llvm::MapVector<Operation *, int64_t> packets;
  int64_t hostWords = 0;
};

struct NativeJob {
  NativeBody *body;
  SmallVector<Value> arguments;
  Value descriptor, frame, empty;
  int64_t thread, entry;
};

// Every scalar occupies one 64-bit word. A memref contributes its allocation
// pointer and DMA base (distinct for aligned allocations),
// dynamic offset, dynamic dimensions and dynamic physical strides.
int64_t frameWords(Type type) {
  if (auto memory = dyn_cast<MemRefType>(type)) {
    auto [strides, offset] = memory.getStridesAndOffset();
    return 2 + memory.getNumDynamicDims() + ShapedType::isDynamic(offset) +
           (memory.getLayout().isIdentity()
                ? 0
                : llvm::count_if(strides, ShapedType::isDynamic));
  }
  if (type.isIndex() || isa<LLVM::LLVMPointerType>(type) ||
      (isa<IntegerType, FloatType>(type) && type.getIntOrFloatBitWidth() <= 64))
    return 1;
  llvm::report_fatal_error(
      "native frame requires a ranked memref or <=64-bit scalar");
}

void packFrame(OpBuilder &b, Location loc, Value frame, Value value,
               int64_t &word) {
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  auto store = [&](Value scalar) {
    if (scalar.getType().isIndex())
      scalar = arith::IndexCastOp::create(b, loc, b.getI64Type(), scalar);
    if (auto floating = dyn_cast<FloatType>(scalar.getType()))
      scalar = arith::BitcastOp::create(
          b, loc, b.getIntegerType(floating.getWidth()), scalar);
    if (auto integer = dyn_cast<IntegerType>(scalar.getType());
        integer && integer.getWidth() < 64)
      scalar = arith::ExtUIOp::create(b, loc, b.getI64Type(), scalar);
    Value address = LLVM::GEPOp::create(
        b, loc, ptr, b.getI64Type(), frame,
        ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(word++)});
    LLVM::StoreOp::create(b, loc, scalar, address);
  };
  if (auto type = dyn_cast<MemRefType>(value.getType())) {
    LLVMTypeConverter converter(b.getContext());
    Value raw = UnrealizedConversionCastOp::create(
                    b, loc, converter.convertType(type), value)
                    .getResult(0);
    store(MemRefDescriptor(raw).allocatedPtr(b, loc));
    auto metadata = memref::ExtractStridedMetadataOp::create(b, loc, value);
    Value address =
        memref::ExtractAlignedPointerAsIndexOp::create(b, loc, value);
    store(LLVM::IntToPtrOp::create(
        b, loc, ptr,
        arith::IndexCastOp::create(b, loc, b.getI64Type(), address)
            .getResult()));
    auto [strides, offset] = type.getStridesAndOffset();
    if (ShapedType::isDynamic(offset))
      store(metadata.getOffset());
    for (auto [axis, size] : llvm::enumerate(type.getShape()))
      if (ShapedType::isDynamic(size))
        store(metadata.getSizes()[axis]);
    if (!type.getLayout().isIdentity())
      for (auto [axis, stride] : llvm::enumerate(strides))
        if (ShapedType::isDynamic(stride))
          store(metadata.getStrides()[axis]);
    return;
  }
  store(value);
}

Value unpackFrame(OpBuilder &b, Location loc, Value frame, Type type,
                  int64_t &word) {
  auto ptr = LLVM::LLVMPointerType::get(b.getContext());
  auto load = [&](Type result) -> Value {
    Value address = LLVM::GEPOp::create(
        b, loc, ptr, b.getI64Type(), frame,
        ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(word++)});
    return LLVM::LoadOp::create(b, loc, result, address);
  };
  if (auto memory = dyn_cast<MemRefType>(type)) {
    LLVMTypeConverter converter(b.getContext());
    auto descriptor =
        MemRefDescriptor::poison(b, loc, converter.convertType(memory));
    Value allocated = load(ptr);
    Value base = load(ptr);
    descriptor.setAllocatedPtr(b, loc, allocated);
    descriptor.setAlignedPtr(b, loc, base);
    auto constant = [&](int64_t n) -> Value {
      return LLVM::ConstantOp::create(b, loc, b.getI64Type(),
                                      b.getI64IntegerAttr(n));
    };
    auto [strides, offset] = memory.getStridesAndOffset();
    descriptor.setOffset(b, loc,
                         ShapedType::isDynamic(offset) ? load(b.getI64Type())
                                                       : constant(offset));
    SmallVector<Value> sizes;
    for (int64_t size : memory.getShape())
      sizes.push_back(ShapedType::isDynamic(size) ? load(b.getI64Type())
                                                  : constant(size));
    for (auto [axis, size] : llvm::enumerate(sizes))
      descriptor.setSize(b, loc, axis, size);
    for (auto [axis, stride] : llvm::enumerate(strides)) {
      Value value;
      if (!ShapedType::isDynamic(stride))
        value = constant(stride);
      else if (!memory.getLayout().isIdentity())
        value = load(b.getI64Type());
      else {
        value = constant(1);
        for (int64_t later = axis + 1; later < memory.getRank(); ++later)
          value = LLVM::MulOp::create(b, loc, value, sizes[later]);
      }
      descriptor.setStride(b, loc, axis, value);
    }
    return UnrealizedConversionCastOp::create(b, loc, memory, Value(descriptor))
        .getResult(0);
  }
  if (isa<LLVM::LLVMPointerType>(type))
    return load(type);
  Value scalar = load(b.getI64Type());
  if (type.isIndex())
    return arith::IndexCastOp::create(b, loc, type, scalar);
  Type bits = isa<FloatType>(type)
                  ? Type(b.getIntegerType(type.getIntOrFloatBitWidth()))
                  : type;
  if (bits.getIntOrFloatBitWidth() < 64)
    scalar = arith::TruncIOp::create(b, loc, bits, scalar);
  if (isa<FloatType>(type))
    scalar = arith::BitcastOp::create(b, loc, type, scalar);
  return scalar;
}

bool custom(Operation *op) {
  auto call = dyn_cast<LLVM::CallIntrinsicOp>(op);
  return call && call.getIntrin() == "llvm.riscv.buckyball.custom";
}

LogicalResult analyzeBody(NativeBody &body) {
  Block *entry = &body.source.getBody().front();
  auto hostValue = [](Operation *op) {
    return isa<memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp,
               memref::LoadOp, func::CallOp>(op);
  };
  std::function<LogicalResult(Value, DenseSet<Operation *> &)> need;
  std::function<LogicalResult(Operation *, DenseSet<Operation *> &)> include;
  need = [&](Value value, DenseSet<Operation *> &selected) -> LogicalResult {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (argument.getOwner() == entry)
        return success();
      return include(argument.getOwner()->getParentOp(), selected);
    }
    Operation *op = value.getDefiningOp();
    if (hostValue(op)) {
      if (isa<memref::AllocaOp>(op))
        return op->emitError("native DMA staging cannot escape a host alloca");
      if (op->getBlock() != entry)
        return op->emitError("native capture must be produced by host "
                             "preparation before the Ant region");
      body.captures.insert(value);
      return success();
    }
    return include(op, selected);
  };
  include = [&](Operation *op,
                DenseSet<Operation *> &selected) -> LogicalResult {
    if (!selected.insert(op).second)
      return success();
    if (!custom(op) && !isa<scf::ForOp, scf::IfOp>(op) &&
        !isMemoryEffectFree(op))
      return op->emitError(
          "native region contains host memory or control effects");
    if (!isa<arith::BitcastOp>(op) &&
        op->getName().getDialectNamespace() == "arith" &&
        llvm::any_of(op->getResultTypes(),
                     [](Type type) { return isa<FloatType>(type); }))
      return op->emitError("native Ant scalar code does not implement "
                           "floating-point arithmetic");
    for (Value operand : op->getOperands())
      if (failed(need(operand, selected)))
        return failure();
    if (op->getParentOp() != body.source)
      if (failed(include(op->getParentOp(), selected)))
        return failure();
    return success();
  };
  bool invalid = false;
  body.source.walk([&](Operation *op) {
    if (custom(op) && failed(include(op, body.ant)))
      invalid = true;
    if (isa<memref::DeallocOp>(op) ||
        (isa<memref::CopyOp>(op) && op->hasAttr("hart.after"))) {
      if (op->getBlock() != entry) {
        op->emitError(
            "native completion currently requires a top-level copy or release");
        invalid = true;
      }
      body.completion.insert(op);
      for (Value operand : op->getOperands())
        if (failed(need(operand, body.completion)))
          invalid = true;
    }
  });
  for (Operation &op : *entry)
    if (body.ant.contains(&op))
      op.walk([&](Operation *nested) {
        if (nested == &op || custom(nested) ||
            nested->hasTrait<OpTrait::IsTerminator>() ||
            isa<scf::ForOp, scf::IfOp>(nested) || isMemoryEffectFree(nested))
          return;
        nested->emitError(
            "host memory effects cannot execute inside an Ant region");
        invalid = true;
      });
  if (invalid)
    return failure();
  body.fields.append(body.source.getArgumentTypes().begin(),
                     body.source.getArgumentTypes().end());
  for (Type type : body.fields)
    body.inputWords += frameWords(type);
  for (Value capture : body.captures)
    body.fields.push_back(capture.getType());
  for (Type type : body.fields)
    body.words += frameWords(type);
  body.hostWords = body.words + 1;
  for (Value capture : body.captures) {
    auto allocation = capture.getDefiningOp<memref::AllocOp>();
    if (!allocation || allocation->getBlock() != entry ||
        allocation.getType() !=
            MemRefType::get({1, 12},
                            IntegerType::get(body.source.getContext(), 64)) ||
        !allocation.getAlignmentAttr() ||
        allocation.getAlignmentAttr().getInt() != 64 ||
        !llvm::any_of(body.completion, [&](Operation *op) {
          auto release = dyn_cast<memref::DeallocOp>(op);
          return release && release.getMemref() == capture;
        }))
      continue;
    int64_t start = llvm::alignTo(body.hostWords * 8, int64_t(64));
    // Preparation receives frame=descriptor+8, not the descriptor base.
    body.packets[allocation] = start / 8 - 1;
    body.hostWords = (start + 96) / 8;
  }
  return success();
}

class LowerHartNativePass
    : public PassWrapper<LowerHartNativePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerHartNativePass)
  LowerHartNativePass() = default;
  LowerHartNativePass(const LowerHartNativePass &pass) : PassWrapper(pass) {}
  StringRef getArgument() const final { return "lower-hart-native"; }
  StringRef getDescription() const final {
    return "Emit native Ant subgraphs, typed frames and existing kernel jobs";
  }
  Option<std::string> designPb{*this, "design", llvm::cl::desc("design.pb")};
  Option<std::string> hartClass{*this, "hart",
                                llvm::cl::desc("hart component")};
  Option<std::string> directory{
      *this, "output-dir", llvm::cl::desc("native program build directory")};
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    if (directory.empty()) {
      module.emitError("lower-hart-native requires its build output-dir");
      return signalPassFailure();
    }
    auto buffer = llvm::MemoryBuffer::getFile(designPb);
    buckyball::config::Design design;
    if (!buffer || !design.ParseFromArray((*buffer)->getBufferStart(),
                                          (*buffer)->getBufferSize())) {
      module.emitError("cannot read native design.pb");
      return signalPassFailure();
    }
    SmallVector<uint64_t> signatures;
    SmallVector<int64_t> codeBytes, tlsBytes;
    int64_t tssBytes = std::numeric_limits<int64_t>::max();
    for (const auto &hart : design.device().harts())
      if (hart.component() == hartClass) {
        tssBytes = std::min(tssBytes, int64_t(hart.tss().bytes()));
        if (!signatures.empty())
          continue;
        for (uint32_t index : hart.thread_indices()) {
          const auto &thread = design.threads(index);
          if (thread.cpu().kind() != "ant") {
            module.emitError("native thread emission requires an Ant CPU");
            return signalPassFailure();
          }
          signatures.push_back(thread.signature());
          codeBytes.push_back(thread.cpu().ant().code_bytes());
          tlsBytes.push_back(thread.cpu().ant().tls().bytes());
        }
      }
    if (signatures.empty()) {
      module.emitError("native hart has no configured threads");
      return signalPassFailure();
    }
    OwningOpRef<ModuleOp> saved = cast<ModuleOp>(module->clone());
    llvm::MapVector<StringRef, std::unique_ptr<NativeBody>> bodies;
    SmallVector<SmallVector<NativeBody *>> threadBodies(signatures.size());
    bool invalid = false;
    module.walk([&](LaunchOp launch) {
      auto name = launch.getCallee();
      if (!bodies.count(name)) {
        auto body = std::make_unique<NativeBody>();
        body->source = saved->lookupSymbol<func::FuncOp>(name);
        if (!body->source || failed(analyzeBody(*body)))
          invalid = true;
        bodies[name] = std::move(body);
      }
      auto *body = bodies[name].get();
      int64_t thread = launch.getThread();
      if (thread < 0 || thread >= int64_t(signatures.size())) {
        launch.emitError("native launch thread is outside the hart");
        invalid = true;
        return;
      }
      if (!llvm::is_contained(threadBodies[thread], body))
        threadBodies[thread].push_back(body);
      module.lookupSymbol<func::FuncOp>(name)->setAttr(
          "hart.native_source", StringAttr::get(context, name));
    });
    if (invalid)
      return signalPassFailure();
    // This is the existing minimum workspace contract, not a proof of the
    // generated machine stack. The native build must also check .stack_sizes.
    constexpr int64_t stackReserve = 1024;
    for (auto [thread, list] : llvm::enumerate(threadBodies))
      for (NativeBody *body : list)
        if (body->words * 8 > tssBytes ||
            body->words * 8 + stackReserve > tlsBytes[thread]) {
          body->source.emitError(
              "native frame exceeds configured TSS or TLS stack reserve");
          return signalPassFailure();
        }
    OpPassManager schedule(ModuleOp::getOperationName());
    schedule.addPass(mlir::buddy::createLowerHartPass(designPb, hartClass));
    if (failed(runPipeline(schedule, module)))
      return signalPassFailure();

    auto ptr = LLVM::LLVMPointerType::get(context);
    auto i64 = IntegerType::get(context, 64);
    auto i32 = IntegerType::get(context, 32);
    auto constant = [&](OpBuilder &b, Location loc, int64_t n) -> Value {
      return arith::ConstantIntOp::create(b, loc, n, 64);
    };
    for (auto &[name, owned] : bodies) {
      NativeBody &body = *owned;
      SmallVector<Type> arguments(body.source.getArgumentTypes());
      arguments.push_back(ptr);
      OpBuilder b = OpBuilder::atBlockEnd(module.getBody());
      Location loc = body.source.getLoc();
      body.prepare =
          func::FuncOp::create(b, loc, (name + "_native_prepare").str(),
                               FunctionType::get(context, arguments, {}));
      body.prepare.setPrivate();
      b.setInsertionPointToStart(body.prepare.addEntryBlock());
      IRMapping map;
      for (auto [from, to] :
           llvm::zip_equal(body.source.getArguments(),
                           body.prepare.getArguments().drop_back()))
        map.map(from, to);
      for (Operation &op : body.source.getBody().front().without_terminator()) {
        auto packet = body.packets.find(&op);
        if (packet == body.packets.end()) {
          b.clone(op, map);
          continue;
        }
        Value memory = LLVM::GEPOp::create(
            b, op.getLoc(), ptr, i64, body.prepare.getArguments().back(),
            ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(packet->second)});
        auto type = cast<MemRefType>(op.getResult(0).getType());
        LLVMTypeConverter converter(context);
        auto descriptor = MemRefDescriptor::fromStaticShape(
            b, op.getLoc(), converter, type, memory);
        map.map(op.getResult(0), UnrealizedConversionCastOp::create(
                                     b, op.getLoc(), type, Value(descriptor))
                                     .getResult(0));
      }
      int64_t word = body.inputWords;
      for (Value capture : body.captures)
        packFrame(b, loc, body.prepare.getArguments().back(),
                  map.lookup(capture), word);
      func::ReturnOp::create(b, loc);
      SmallVector<Operation *> erase;
      body.prepare.walk<WalkOrder::PostOrder>([&](Operation *op) {
        if (custom(op) || isa<memref::DeallocOp>(op) ||
            (isa<memref::CopyOp>(op) && op->hasAttr("hart.after")))
          erase.push_back(op);
      });
      for (Operation *op : erase)
        op->erase();

      b.setInsertionPointToEnd(module.getBody());
      body.complete =
          func::FuncOp::create(b, loc, (name + "_native_complete").str(),
                               FunctionType::get(context, arguments, {}));
      body.complete.setPrivate();
      b.setInsertionPointToStart(body.complete.addEntryBlock());
      map.clear();
      for (auto [from, to] :
           llvm::zip_equal(body.source.getArguments(),
                           body.complete.getArguments().drop_back()))
        map.map(from, to);
      word = body.inputWords;
      for (Value capture : body.captures)
        map.map(capture,
                unpackFrame(b, loc, body.complete.getArguments().back(),
                            capture.getType(), word));
      for (Operation &op : body.source.getBody().front().without_terminator()) {
        auto release = dyn_cast<memref::DeallocOp>(op);
        if (release && body.packets.count(release.getMemref().getDefiningOp()))
          continue;
        if (body.completion.contains(&op))
          b.clone(op, map)->removeAttr("hart.after");
      }
      func::ReturnOp::create(b, loc);
    }

    // Inline only the generated scalar-frame launchers. Their allocas and
    // descriptors then belong to the batch parent through wait/completion.
    DenseMap<Operation *, NativeJob> jobs;
    SmallVector<func::CallOp> launchers;
    module.walk([&](func::CallOp call) {
      if (call.getCallee().contains("_launch_"))
        launchers.push_back(call);
    });
    for (func::CallOp call : launchers) {
      auto helper = module.lookupSymbol<func::FuncOp>(call.getCallee());
      StringRef sourceName;
      for (auto &[name, body] : bodies)
        if (call.getCallee().starts_with((name + "_launch_").str()))
          sourceName = name;
      if (sourceName.empty()) {
        call.emitError("native launcher has no compiled body");
        return signalPassFailure();
      }
      NativeBody *body = bodies[sourceName].get();
      IRMapping map;
      for (auto [from, to] :
           llvm::zip_equal(helper.getArguments(), call.getOperands()))
        map.map(from, to);
      OpBuilder b(call);
      for (Operation &op : helper.getBody().front().without_terminator()) {
        Operation *cloned = b.clone(op, map);
        auto job = dyn_cast<func::CallOp>(cloned);
        if (!job || job.getCallee() != "stream_job")
          continue;
        APInt index;
        if (!matchPattern(job.getOperand(1), m_ConstantInt(&index))) {
          job.emitError("native placement must identify a physical thread");
          return signalPassFailure();
        }
        int64_t thread = index.getSExtValue();
        int64_t entry = llvm::find(threadBodies[thread], body) -
                        threadBodies[thread].begin();
        Value oldFrame = job.getOperand(3);
        auto allocation = oldFrame.getDefiningOp<LLVM::AllocaOp>();
        OpBuilder frameBuilder(allocation);
        Value descriptor = LLVM::AllocaOp::create(
            frameBuilder, call.getLoc(), ptr, i64,
            constant(frameBuilder, call.getLoc(), body->hostWords));
        if (!body->packets.empty())
          descriptor.getDefiningOp()->setAttr(
              "alignment", frameBuilder.getI64IntegerAttr(64));
        Value frame =
            LLVM::GEPOp::create(frameBuilder, call.getLoc(), ptr, i64,
                                descriptor, ArrayRef<LLVM::GEPArg>{1});
        LLVM::StoreOp::create(frameBuilder, call.getLoc(),
                              constant(frameBuilder, call.getLoc(), entry),
                              descriptor);
        oldFrame.replaceAllUsesWith(frame);
        allocation.erase();
        jobs[job] = NativeJob{body,
                              SmallVector<Value>(call.getOperands()),
                              descriptor,
                              frame,
                              job.getOperand(4),
                              thread,
                              entry};
      }
      call.erase();
    }

    auto declare = [&](StringRef name, TypeRange inputs, TypeRange outputs) {
      if (auto fn = module.lookupSymbol<func::FuncOp>(name))
        return fn;
      OpBuilder b = OpBuilder::atBlockEnd(module.getBody());
      auto fn =
          func::FuncOp::create(b, module.getLoc(), name,
                               FunctionType::get(context, inputs, outputs));
      fn.setPrivate();
      return fn;
    };
    auto current = declare("kernel_current", {}, {ptr});
    auto begin = declare("kernel_begin", {ptr, ptr, i64, ptr, i64, ptr}, {});
    auto publish = declare("kernel_publish", {ptr, i64}, {});
    auto wait = declare("kernel_wait", {ptr}, {});
    auto end = declare("kernel_end", {ptr}, {});
    auto taskType = LLVM::LLVMStructType::getLiteral(
        context, {i64, i64, i64, i64, i64, i64});
    auto jobType = LLVM::LLVMStructType::getLiteral(
        context, {i32, taskType, ptr, i64, ptr, i32});
    auto resultType =
        LLVM::LLVMStructType::getLiteral(context, {i32, i32, i64});
    auto transferType = LLVM::LLVMStructType::getLiteral(
        context, {i64, i64, i64, i64, i64, i64});
    auto address = [&](OpBuilder &b, Location loc, Type type, Value base,
                       ArrayRef<LLVM::GEPArg> indices) -> Value {
      return LLVM::GEPOp::create(b, loc, ptr, type, base, indices);
    };
    auto store = [&](OpBuilder &b, Location loc, Value value, Type type,
                     Value base, ArrayRef<LLVM::GEPArg> indices) {
      LLVM::StoreOp::create(b, loc, value,
                            address(b, loc, type, base, indices));
    };
    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    for (auto function : functions) {
      if (function.isExternal())
        continue;
      Value kernel, jobArray, results, dependencies, transfers;
      int64_t count = 0, dependencyCount = 0, transferCount = 0;
      int64_t nextJob = 0, nextDependency = 0, nextTransfer = 0;
      SmallVector<NativeJob *> completion;
      SmallVector<func::CallOp> calls;
      for (auto call : function.getBody().front().getOps<func::CallOp>())
        if (call.getCallee().starts_with("stream_"))
          calls.push_back(call);
      for (auto call : calls) {
        if (!kernel && call.getCallee() != "stream_begin")
          continue;
        OpBuilder b(call);
        Location loc = call.getLoc();
        auto integer = [&](Value value) -> int64_t {
          APInt bits;
          if (!matchPattern(value, m_ConstantInt(&bits)))
            llvm::report_fatal_error(
                "native batch size must be a compile-time constant");
          return bits.getSExtValue();
        };
        if (call.getCallee() == "stream_begin") {
          count = integer(call.getOperand(0));
          dependencyCount = integer(call.getOperand(1));
          transferCount = integer(call.getOperand(2));
          nextJob = nextDependency = nextTransfer = 0;
          completion.clear();
          kernel =
              func::CallOp::create(b, loc, current, ValueRange{}).getResult(0);
          jobArray = LLVM::AllocaOp::create(b, loc, ptr, jobType,
                                            constant(b, loc, count));
          results = LLVM::AllocaOp::create(b, loc, ptr, resultType,
                                           constant(b, loc, count));
          dependencies = LLVM::AllocaOp::create(
              b, loc, ptr, i64, constant(b, loc, dependencyCount));
          transfers = LLVM::AllocaOp::create(b, loc, ptr, transferType,
                                             constant(b, loc, transferCount));
          if (!transferCount)
            func::CallOp::create(b, loc, begin,
                                 ValueRange{kernel, jobArray,
                                            constant(b, loc, 0), transfers,
                                            constant(b, loc, 0), results});
        } else if (call.getCallee() == "stream_job") {
          auto found = jobs.find(call);
          if (found == jobs.end()) {
            call.emitError("uncompiled stream callback in native batch");
            return signalPassFailure();
          }
          NativeJob &job = found->second;
          if (!transferCount)
            func::CallOp::create(b, loc, publish,
                                 ValueRange{kernel, constant(b, loc, nextJob)});
          Value nonempty = arith::CmpIOp::create(
              b, loc, arith::CmpIPredicate::eq, job.empty,
              arith::ConstantIntOp::create(b, loc, 0, 32));
          auto prepare = scf::IfOp::create(b, loc, nonempty, false);
          OpBuilder prep =
              OpBuilder::atBlockBegin(&prepare.getThenRegion().front());
          SmallVector<Value> arguments(job.arguments);
          arguments.push_back(job.frame);
          func::CallOp::create(prep, loc, job.body->prepare, arguments);
          auto index = static_cast<int32_t>(nextJob);
          store(b, loc, call.getOperand(1), jobType, jobArray, {index, 0});
          for (int32_t field = 0; field < 6; ++field)
            store(b, loc,
                  field == 0   ? constant(b, loc, nextJob + 1)
                  : field == 5 ? call.getOperand(0)
                               : constant(b, loc, 0),
                  jobType, jobArray, {index, 1, field});
          store(b, loc,
                address(b, loc, i64, dependencies,
                        {static_cast<int32_t>(nextDependency)}),
                jobType, jobArray, {index, 2});
          store(b, loc, constant(b, loc, 0), jobType, jobArray, {index, 3});
          store(b, loc, job.descriptor, jobType, jobArray, {index, 4});
          store(b, loc, job.empty, jobType, jobArray, {index, 5});
          completion.push_back(&job);
          ++nextJob;
        } else if (call.getCallee() == "stream_after") {
          store(b, loc, call.getOperand(0), i64, dependencies,
                {static_cast<int32_t>(nextDependency++)});
          Value sizeAddress = address(b, loc, jobType, jobArray,
                                      {static_cast<int32_t>(nextJob - 1), 3});
          Value size = LLVM::LoadOp::create(b, loc, i64, sizeAddress);
          LLVM::StoreOp::create(
              b, loc, arith::AddIOp::create(b, loc, size, constant(b, loc, 1)),
              sizeAddress);
        } else if (call.getCallee() == "stream_transfer") {
          for (int32_t field = 0; field < 6; ++field) {
            Value value =
                field == 5 ? constant(b, loc, -1) : call.getOperand(field);
            if (isa<LLVM::LLVMPointerType>(value.getType()))
              value = LLVM::PtrToIntOp::create(b, loc, i64, value);
            store(b, loc, value, transferType, transfers,
                  {static_cast<int32_t>(nextTransfer), field});
          }
          ++nextTransfer;
        } else if (call.getCallee() == "stream_submit") {
          if (nextJob != count || nextDependency != dependencyCount ||
              nextTransfer != transferCount) {
            call.emitError("native batch metadata differs from scheduled jobs");
            return signalPassFailure();
          }
          if (transferCount)
            func::CallOp::create(
                b, loc, begin,
                ValueRange{kernel, jobArray, constant(b, loc, count), transfers,
                           constant(b, loc, transferCount), results});
          else
            func::CallOp::create(b, loc, publish,
                                 ValueRange{kernel, constant(b, loc, count)});
        } else if (call.getCallee() == "stream_wait") {
          func::CallOp::create(b, loc, wait, ValueRange{kernel});
          for (int32_t index = 0; index < count; ++index) {
            Value state = LLVM::LoadOp::create(
                b, loc, i32, address(b, loc, resultType, results, {index, 0}));
            Value failed = arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate::ne, state,
                arith::ConstantIntOp::create(b, loc, 3, 32));
            auto rejected = scf::IfOp::create(b, loc, failed, false);
            OpBuilder trap =
                OpBuilder::atBlockBegin(&rejected.getThenRegion().front());
            LLVM::CallIntrinsicOp::create(
                trap, loc, trap.getStringAttr("llvm.trap"), ValueRange{});
          }
          func::CallOp::create(b, loc, end, ValueRange{kernel});
          for (NativeJob *job : llvm::reverse(completion)) {
            Value nonempty = arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate::eq, job->empty,
                arith::ConstantIntOp::create(b, loc, 0, 32));
            auto complete = scf::IfOp::create(b, loc, nonempty, false);
            OpBuilder done =
                OpBuilder::atBlockBegin(&complete.getThenRegion().front());
            SmallVector<Value> arguments(job->arguments);
            arguments.push_back(job->frame);
            func::CallOp::create(done, loc, job->body->complete, arguments);
          }
        } else {
          // Only callers owning a native batch are rewritten. Private old
          // recorder functions disappear with symbol DCE below.
          if (!kernel)
            continue;
          call.emitError("unsupported operation in native batch");
          return signalPassFailure();
        }
        call.erase();
      }
    }

    std::error_code error = llvm::sys::fs::create_directories(directory);
    if (error) {
      module.emitError("cannot create native program build directory");
      return signalPassFailure();
    }
    auto output = [&](StringRef filename) {
      llvm::SmallString<256> path(directory);
      llvm::sys::path::append(path, filename);
      auto stream = std::make_unique<llvm::raw_fd_ostream>(path, error);
      if (error)
        llvm::report_fatal_error(llvm::Twine("cannot write native artifact ") +
                                 path);
      return stream;
    };
    llvm::json::Array manifest;
    for (auto [thread, list] : llvm::enumerate(threadBodies)) {
      if (list.empty())
        continue;
      OwningOpRef<ModuleOp> program = ModuleOp::create(module.getLoc());
      OpBuilder b = OpBuilder::atBlockEnd(program->getBody());
      llvm::json::Array entries;
      auto assembly = output(("thread-" + llvm::Twine(thread) + ".S").str());
      *assembly << ".option norvc\n.option norelax\n.section "
                   ".text.start,\"ax\",@progbits\n.globl _start\n_start:\n";
      for (auto [index, body] : llvm::enumerate(list)) {
        Location loc = body->source.getLoc();
        auto function =
            func::FuncOp::create(b, loc, ("native_" + llvm::Twine(index)).str(),
                                 FunctionType::get(context, body->fields, {}));
        function.setPrivate();
        function->setAttr(
            "llvm.linkage",
            LLVM::LinkageAttr::get(context, LLVM::Linkage::Internal));
        b.setInsertionPointToStart(function.addEntryBlock());
        IRMapping map;
        unsigned argument = 0;
        for (Value value : body->source.getArguments())
          map.map(value, function.getArgument(argument++));
        for (Value value : body->captures)
          map.map(value, function.getArgument(argument++));
        for (Operation &op :
             body->source.getBody().front().without_terminator())
          if (body->ant.contains(&op))
            b.clone(op, map);
        func::ReturnOp::create(b, loc);
        b.setInsertionPointToEnd(program->getBody());
        auto entryName = ("bb_native_entry_" + llvm::Twine(index)).str();
        auto entry = func::FuncOp::create(
            b, loc, entryName, FunctionType::get(context, {ptr}, {}));
        entry->setAttr("buckyball.public_symbol", b.getStringAttr(entryName));
        b.setInsertionPointToStart(entry.addEntryBlock());
        int64_t word = 0;
        SmallVector<Value> values;
        for (Type type : body->fields)
          values.push_back(
              unpackFrame(b, loc, entry.getArgument(0), type, word));
        func::CallOp::create(b, loc, function, values);
        func::ReturnOp::create(b, loc);
        b.setInsertionPointToEnd(program->getBody());
        *assembly
            << ".balign 4\n.globl bb_native_start_" << index
            << "\nbb_native_start_" << index << ":\n  call " << entryName
            << "\n  fence\n  li a0, 0\n  li a7, 0\n  ecall\n  ebreak\n  nop\n";
        entries.push_back(
            llvm::json::Object{{"source", body->source.getName().str()},
                               {"ordinal", int64_t(index)},
                               {"entry", int64_t(index * 32)},
                               {"frame_bytes", body->words * 8}});
      }
      PassManager cleanup(context);
      cleanup.addPass(createCanonicalizerPass());
      cleanup.addPass(createCSEPass());
      cleanup.addPass(func::createDuplicateFunctionEliminationPass());
      if (failed(cleanup.run(*program)))
        return signalPassFailure();
      auto stream = output(("thread-" + llvm::Twine(thread) + ".mlir").str());
      program->print(*stream);
      manifest.push_back(
          llvm::json::Object{{"thread", int64_t(thread)},
                             {"signature", llvm::utohexstr(signatures[thread])},
                             {"code_bytes", codeBytes[thread]},
                             {"tls_bytes", tlsBytes[thread]},
                             {"tss_bytes", tssBytes},
                             {"stack_reserve_bytes", stackReserve},
                             {"entries", std::move(entries)}});
    }
    auto loader = output("native-bindings.c");
    *loader << "#include <kernel.h>\n";
    for (auto [thread, list] : llvm::enumerate(threadBodies))
      if (!list.empty())
        *loader << "static const unsigned char image_" << thread
                << "[] __attribute__((aligned(8))) = {\n#embed \"thread-"
                << thread << ".bin\"\n};\n_Static_assert(((sizeof(image_"
                << thread << ")+7)&~(size_t)7)<=" << codeBytes[thread]
                << ",\"native program exceeds configured code storage\");\n";
    *loader << "struct native_arguments { uint64_t entry, frame[]; };\n";
    for (uint64_t signature :
         llvm::SetVector<uint64_t>(signatures.begin(), signatures.end())) {
      std::string id = llvm::utohexstr(signature);
      *loader
          << "static void load_" << id << "(uint32_t thread) {\n"
          << " const struct kernel_slot "
             "*slot=&kernel_current()->slots[thread];\n"
          << " const unsigned char *image; size_t bytes;\n switch(thread) {\n";
      for (auto [thread, list] : llvm::enumerate(threadBodies))
        if (signatures[thread] == signature && !list.empty())
          *loader << " case " << thread << ": image=image_" << thread
                  << "; bytes=sizeof(image_" << thread << "); break;\n";
      *loader
          << " default: __builtin_trap();\n }\n"
          << " if (((bytes+7)&~(size_t)7)>slot->code_bytes) __builtin_trap();\n"
          << " for (size_t offset=0;offset<bytes;offset+=slot->tss_bytes) {\n"
          << "  size_t count=bytes-offset; if(count>slot->tss_bytes) "
             "count=slot->tss_bytes;\n"
          << "  thread_tss_write(0,image+offset,count);\n"
          << "  "
             "thread_load(thread,THREAD_CODE,offset,0,(count+7)&~(size_t)7);\n "
             "}\n}\n";
      *loader << "static void launch_" << id
              << "(uint32_t thread,void *opaque,struct thread_task *task) {\n"
              << " const struct kernel_slot "
                 "*slot=&kernel_current()->slots[thread];\n"
              << " const struct native_arguments *arguments=opaque; size_t "
                 "bytes;\n switch(thread) {\n";
      for (auto [thread, list] : llvm::enumerate(threadBodies))
        if (signatures[thread] == signature && !list.empty()) {
          *loader << " case " << thread << ": task->code_end=sizeof(image_"
                  << thread << "); switch(arguments->entry) {\n";
          for (auto [index, body] : llvm::enumerate(list))
            *loader << " case " << index << ": bytes=" << body->words * 8
                    << "; break;\n";
          *loader << " default: __builtin_trap();\n } break;\n";
        }
      *loader
          << " default: __builtin_trap();\n }\n"
          << " task->entry=32*arguments->entry; "
             "task->argument=slot->tls_base;\n"
          << " task->stack=slot->tls_base+slot->tls_bytes;\n"
          << " if(bytes) "
             "thread_start_data(thread,task,arguments->frame,bytes);\n"
          << " else thread_start(thread,task);\n}\n"
          << "static const struct kernel_binding binding_" << id
          << " __attribute__((section(\"kernel_bindings\"),used))={UINT64_C(0x"
          << id << "),load_" << id << ",launch_" << id << "};\n";
    }
    *output("native.json") << llvm::formatv(
        "{0:2}\n", llvm::json::Value(std::move(manifest)));
    OpPassManager cleanup(ModuleOp::getOperationName());
    cleanup.addPass(createCanonicalizerPass());
    cleanup.addPass(createCSEPass());
    cleanup.addPass(createSymbolDCEPass());
    if (failed(runPipeline(cleanup, module)))
      return signalPassFailure();
  }
};
PassRegistration<LowerHartNativePass> registration;
} // namespace

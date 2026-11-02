//===- LowerDevice.cpp - A placed program on the device runtime -----------===//
//
// lower-device lowers the bufferized launches and transfers of a program,
// placed by the design's convert-linalg-to-device, to the job runtime of the
// framework (stack/runtime/program) and the device runtime of the design,
// which provides the transfers between its cores:
//
//   launch       program_job_begin(core, entry), program_job_argument(word)
//                for the address and the dynamic sizes of every buffer,
//                program_job_after(job) for every job that writes one of its
//                inputs, then program_job_submit() -> job
//   transfer     of a value a job writes and only jobs read:
//                device_transfer(from, to, target, source, bytes, job) -> job,
//                after which `target` holds the value; of a value the
//                program made, or one the program reads itself: the program
//                waits for the job that wrote it, if any, and moves it now,
//                device_copy(from, to, target, source, bytes)
//   memref.copy  of a buffer a job writes (a result the program returns):
//                program_copy(core, target, source, bytes, job) -> job, a
//                copy on the program's core after `job`
//
// How a value crosses from one core to another (a copy over shared DDR, a
// link between the cores) is the design's: its runtime library provides
// device_transfer and device_copy, those of them its programs call.
//
// A job runs `entry(core context, words)`. The entry of a graph,
// <graph>_job, rebuilds the memrefs of the graph from the words and calls it;
// the program gets one as well, <program>_job, which the program's core runs
// as the first job of the device. The program waits only for the values it
// reads: the jobs of every core, its own included, run in the order their
// dependencies allow.
//
// Every buffer a job reads or writes is contiguous; every buffer a launch or
// a transfer writes is a buffer of its own (an allocation or an argument of
// the program), so no job writes a buffer another job reads before.
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

using namespace mlir;
using namespace ::buddy::device;

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
    func::FuncOp begin = declare("program_job_begin", {i64, entryType}, {});
    func::FuncOp argument = declare("program_job_argument", {i64}, {});
    func::FuncOp after = declare("program_job_after", {i64}, {});
    func::FuncOp submit = declare("program_job_submit", {}, {i64});
    func::FuncOp wait = declare("program_wait", {i64}, {});
    func::FuncOp cleanup = declare("program_cleanup_begin", {i64}, {});
    func::FuncOp copy =
        declare("program_copy", {i64, i64, i64, i64, i64}, {i64});
    func::FuncOp transfer =
        declare("device_transfer", {i64, i64, i64, i64, i64, i64}, {i64});
    func::FuncOp move = declare("device_copy", {i64, i64, i64, i64, i64}, {});

    // <function>_job(core context, words): for every memref argument of
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
        if (!memref.getLayout().isIdentity())
          llvm::report_fatal_error(llvm::Twine("argument of ") +
                                   function.getName() +
                                   " without the identity layout");
        Value pointer = next();
        Value descriptor;
        if (memref.hasStaticShape()) {
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

    auto contiguous = [](Value buffer) {
      auto type = cast<MemRefType>(buffer.getType());
      if (type.getLayout().isIdentity())
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
    // A buffer each launch and transfer writes: its own.
    auto own = [](Value buffer) {
      while (auto view =
                 dyn_cast_or_null<ViewLikeOpInterface>(buffer.getDefiningOp()))
        buffer = view.getViewSource();
      if (!isa<BlockArgument>(buffer) &&
          !buffer.getDefiningOp<memref::AllocOp>())
        llvm::report_fatal_error("a job writes a buffer that is not its own");
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
          program->getAttrOfType<IntegerAttr>("device.core").getInt();
      program->removeAttr("device.core");
      // The job that writes each buffer.
      DenseMap<Value, Value> writers;
      for (Operation &op : llvm::make_early_inc_range(program.getOps())) {
        OpBuilder builder(&op);
        Location loc = op.getLoc();
        if (auto launch = dyn_cast<LaunchOp>(op)) {
          auto callee = module.lookupSymbol<func::FuncOp>(launch.getCallee());
          Value job = func::ConstantOp::create(builder, loc, entryType,
                                               entry(callee).getName());
          func::CallOp::create(
              builder, loc, begin,
              ValueRange{constant(builder, loc, *launch.getCore()), job});
          for (Value buffer : launch->getOperands()) {
            func::CallOp::create(builder, loc, argument,
                                 ValueRange{address(builder, loc, buffer)});
            for (Value size : sizes(builder, loc, buffer))
              func::CallOp::create(builder, loc, argument, ValueRange{size});
          }
          SetVector<Value> waited;
          for (Value input : launch.getInputs())
            if (Value writer = writers.lookup(input))
              waited.insert(writer);
          for (Value writer : waited)
            func::CallOp::create(builder, loc, after, ValueRange{writer});
          Value submitted =
              func::CallOp::create(builder, loc, submit, ValueRange{})
                  .getResult(0);
          for (Value out : launch.getOuts()) {
            own(out);
            writers[out] = submitted;
          }
          launch.erase();
          continue;
        }
        if (auto moved = dyn_cast<TransferOp>(op)) {
          Value writer = writers.lookup(moved.getSource());
          Value from = constant(builder, loc, moved.getFrom());
          Value to = constant(builder, loc, moved.getTo());
          Value target = address(builder, loc, moved.getTarget());
          Value source = address(builder, loc, moved.getSource());
          Value size = bytes(builder, loc, moved.getSource());
          if (writer && !read(moved.getTarget(), moved)) {
            own(moved.getTarget());
            writers[moved.getTarget()] =
                func::CallOp::create(
                    builder, loc, transfer,
                    ValueRange{from, to, target, source, size, writer})
                    .getResult(0);
          } else {
            if (writer)
              func::CallOp::create(builder, loc, wait, ValueRange{writer});
            func::CallOp::create(builder, loc, move,
                                 ValueRange{from, to, target, source, size});
          }
          moved.erase();
          continue;
        }
        if (auto view = dyn_cast<ViewLikeOpInterface>(op)) {
          if (Value writer = writers.lookup(view.getViewSource()))
            writers[op.getResult(0)] = writer;
          continue;
        }
        if (auto copied = dyn_cast<memref::CopyOp>(op)) {
          if (copied.getSource() == copied.getTarget()) {
            copied.erase();
            continue;
          }
          Value writer = writers.lookup(copied.getSource());
          if (!writer)
            continue;
          own(copied.getTarget());
          writers[copied.getTarget()] =
              func::CallOp::create(
                  builder, loc, copy,
                  ValueRange{constant(builder, loc, control),
                             address(builder, loc, copied.getTarget()),
                             address(builder, loc, copied.getSource()),
                             bytes(builder, loc, copied.getSource()), writer})
                  .getResult(0);
          copied.erase();
          continue;
        }
        // Sizes and other metadata of a buffer do not read it.
        if (isMemoryEffectFree(&op))
          continue;
        for (Value operand : op.getOperands())
          if (writers.contains(operand))
            llvm::report_fatal_error(
                llvm::Twine("the program reads a value of a job in ") +
                op.getName().getStringRef());
      }
      // Intermediate buffers outlive the program's submission job. Free
      // them in a final job after the consumers on every core have finished.
      SmallVector<Value> allocations;
      for (auto allocation : program.getOps<memref::AllocOp>())
        allocations.push_back(allocation.getResult());
      if (!allocations.empty()) {
        OpBuilder builder(program.getBody().front().getTerminator());
        Location loc = program.getLoc();
        func::CallOp::create(
            builder, loc, cleanup,
            ValueRange{constant(builder, loc, allocations.size())});
        for (Value buffer : allocations) {
          Value descriptor =
              UnrealizedConversionCastOp::create(
                  builder, loc, converter.convertType(buffer.getType()), buffer)
                  .getResult(0);
          Value pointer =
              MemRefDescriptor(descriptor).allocatedPtr(builder, loc);
          Value word = LLVM::PtrToIntOp::create(builder, loc, i64, pointer);
          func::CallOp::create(builder, loc, argument, ValueRange{word});
        }
        func::CallOp::create(builder, loc, submit, ValueRange{});
      }
      entry(program);
    }
  }
};

PassRegistration<LowerDevicePass> registration;
} // namespace

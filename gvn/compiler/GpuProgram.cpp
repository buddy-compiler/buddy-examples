//===- GpuProgram.cpp - Record the slice values of a device program -------===//
//
// gvn-gpu-program{values=<json> layout=<json> manifest=<file>}, on device
// programs at the tensor level with the linalg definitions of their graphs
// (gvn-import), for the GPU: each program (a public function launching
// graphs, marked gvn.program for gvn-packed-entry) becomes host code calling
// its graphs, and the values the slices read and write are recorded as they
// are computed. The parameter packs of a program are its leading static
// one-dimensional arguments.
//
// A transfer is its source. A launch, the n-th of the programs in their
// order (each numbered after those of the program before), becomes
// gvn_launch(n), then the call of its graph with its inputs and its
// destinations; each operand is recorded (gvn_record, kind 0, its index)
// unless it is an argument of the program or a static slice of one (a
// parameter pack), which the layout gives instead; each result is recorded
// (kind 2). A graph that returns its results is called with its inputs
// only.
// In a graph, the value of each operation located at an id of `values` (for
// that graph) is recorded after it (kind 1, the number of the id). The host
// runtime of the GPU (gvn/gpu.cpp of the model recipes) defines gvn_launch and
// gvn_record_<type>, which take the value as an unranked memref.
//
// `values` maps each graph to its ids; the layout lists the launches with
// their operands, the numbers of the ids, and the ids at which the graph has
// no value (unrecorded), each launch with its program. The manifest, for the
// host runtime, gives for each program its name, its argument count and its
// count of packs, then the ranks of its results, then each argument's element
// bytes, rank and sizes (-1 dynamic), one line each; the runtime calls a
// program <p> through gvn_<p> (gvn-packed-entry).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
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

std::optional<std::string> locationId(Operation *op) {
  Location loc = op->getLoc();
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
    return std::nullopt;
  return std::to_string(file.getLine()) + ":" +
         std::to_string(file.getColumn());
}

class GpuProgramPass
    : public PassWrapper<GpuProgramPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuProgramPass)
  GpuProgramPass() = default;
  GpuProgramPass(const GpuProgramPass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "gvn-gpu-program"; }
  StringRef getDescription() const final {
    return "Make a device program host code recording the slice values";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  Option<std::string> values{*this, "values",
                             llvm::cl::desc("The ids of each graph")};
  Option<std::string> layout{*this, "layout",
                             llvm::cl::desc("The layout file to write")};
  Option<std::string> manifest{*this, "manifest",
                               llvm::cl::desc("The manifest file to write")};

  ModuleOp module;

  // Records `value` with `kind` and `index` after the builder's point.
  void record(OpBuilder &builder, Location loc, int64_t kind, int64_t index,
              Value value) {
    auto type = cast<RankedTensorType>(value.getType());
    MLIRContext *context = module.getContext();
    std::string name = "gvn_record_" + elementName(type.getElementType());
    auto unranked = UnrankedMemRefType::get(type.getElementType(), 0);
    if (!module.lookupSymbol<func::FuncOp>(name)) {
      OpBuilder top(context);
      top.setInsertionPointToStart(module.getBody());
      func::FuncOp::create(
          top, loc, name,
          top.getFunctionType({top.getI64Type(), top.getI64Type(), unranked},
                              {}))
          .setPrivate();
    }
    SmallVector<int64_t> strides(type.getRank(), ShapedType::kDynamic);
    auto buffer = bufferization::ToBufferOp::create(
        builder, loc,
        MemRefType::get(
            type.getShape(), type.getElementType(),
            StridedLayoutAttr::get(context, ShapedType::kDynamic, strides)),
        value, true);
    Value cast = memref::CastOp::create(builder, loc, unranked, buffer);
    func::CallOp::create(
        builder, loc, name, TypeRange{},
        ValueRange{arith::ConstantIntOp::create(builder, loc, kind, 64),
                   arith::ConstantIntOp::create(builder, loc, index, 64),
                   cast});
  }

  void runOnOperation() override {
    module = getOperation();
    MLIRContext *context = module.getContext();
    // The argument layouts are the NPU's zero-copy views (a few heads of a
    // cache); the GPU computes the values in buffers of its own.
    module.walk([](func::FuncOp function) {
      for (unsigned argument = 0; argument < function.getNumArguments();
           ++argument)
        function.removeArgAttr(argument, "bufferization.buffer_layout");
    });
    auto buffer = llvm::MemoryBuffer::getFile(values);
    if (!buffer)
      llvm::report_fatal_error("gvn-gpu-program: cannot read the values");
    auto parsed = llvm::json::parse((*buffer)->getBuffer());
    if (!parsed)
      llvm::report_fatal_error("gvn-gpu-program: malformed values");

    // The values of the graphs.
    llvm::json::Array numbers, unrecorded;
    int64_t number = 0;
    for (auto &[graph, ids] : *parsed->getAsObject()) {
      auto function = module.lookupSymbol<func::FuncOp>(graph);
      if (!function || function.isExternal())
        llvm::report_fatal_error(llvm::Twine("gvn-gpu-program: no graph ") +
                                 graph.str());
      // The graph and the functions it calls (a public graph calling its
      // body, as the recipe outlines it), where its values are.
      SmallVector<func::FuncOp> bodies{function};
      for (size_t at = 0; at < bodies.size(); ++at)
        bodies[at].walk([&](func::CallOp call) {
          auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
          if (!callee.isExternal() && !llvm::is_contained(bodies, callee))
            bodies.push_back(callee);
        });
      for (const llvm::json::Value &id : *ids.getAsArray()) {
        Operation *found = nullptr;
        for (func::FuncOp body : bodies)
          for (Operation &op : body.front().without_terminator())
            if (locationId(&op) == id.getAsString()->str())
              found = &op;
        // A location that names no value of the graph (a shared thread
        // function's, a compiler-made operand's) is listed, not recorded.
        if (!found || found->getNumResults() != 1) {
          unrecorded.push_back(llvm::json::Object{{"graph", graph.str()},
                                                  {"id", *id.getAsString()}});
          continue;
        }
        OpBuilder builder(context);
        builder.setInsertionPointAfter(found);
        record(builder, found->getLoc(), 1, number, found->getResult(0));
        numbers.push_back(llvm::json::Object{{"graph", graph.str()},
                                             {"id", *id.getAsString()},
                                             {"number", number}});
        ++number;
      }
    }

    OpBuilder builder(context);
    builder.setInsertionPointToStart(module.getBody());
    auto launched = func::FuncOp::create(
        builder, module.getLoc(), "gvn_launch",
        builder.getFunctionType({builder.getI64Type()}, {}));
    launched.setPrivate();

    SmallVector<func::FuncOp> programs;
    for (auto function : module.getOps<func::FuncOp>())
      if (function.isPublic() && function
                                     .walk([](Operation *op) {
                                       return op->getName().getStringRef() ==
                                                      "device.launch"
                                                  ? WalkResult::interrupt()
                                                  : WalkResult::advance();
                                     })
                                     .wasInterrupted())
        programs.push_back(function);
    std::error_code error;
    llvm::raw_fd_ostream text(manifest, error);
    if (error)
      llvm::report_fatal_error("gvn-gpu-program: cannot write the manifest");
    llvm::json::Array launches;
    int64_t index = 0;
    for (func::FuncOp program : programs) {
      std::string name = program.getSymName().str();
      program->setAttr("gvn.program", builder.getUnitAttr());
      int64_t packCount = 0;
      for (Type type : program.getArgumentTypes()) {
        auto tensor = cast<RankedTensorType>(type);
        if (tensor.getRank() != 1 || !tensor.hasStaticShape())
          break;
        ++packCount;
      }
      SmallVector<Operation *> ops;
      program.walk([&](Operation *op) {
        StringRef name = op->getName().getStringRef();
        if (name == "device.transfer" || name == "device.launch")
          ops.push_back(op);
      });
      for (Operation *op : ops) {
        Location loc = op->getLoc();
        if (op->getName().getStringRef() == "device.transfer") {
          op->getResult(0).replaceAllUsesWith(op->getOperand(0));
          op->erase();
          continue;
        }
        auto destination = cast<DestinationStyleOpInterface>(op);
        auto callee = module.lookupSymbol<func::FuncOp>(
            op->getAttrOfType<FlatSymbolRefAttr>("callee"));
        // A graph that returns its results takes no destinations.
        SmallVector<Value> operands(destination.getDpsInputs());
        if (callee.getNumArguments() != operands.size())
          for (Value init : destination.getDpsInits())
            operands.push_back(init);
        builder.setInsertionPoint(op);
        func::CallOp::create(
            builder, loc, launched,
            ValueRange{arith::ConstantIntOp::create(builder, loc, index, 64)});
        llvm::json::Array described;
        for (auto [position, operand] : llvm::enumerate(operands)) {
          // The first `packs` program arguments are the immutable parameter
          // packs (F32 and INT8 ones).
          auto whole = dyn_cast<BlockArgument>(operand);
          auto wholeType = dyn_cast<RankedTensorType>(operand.getType());
          if (whole && whole.getOwner() == &program.front() &&
              whole.getArgNumber() < packCount && wholeType.getRank() == 1 &&
              wholeType.hasStaticShape()) {
            described.push_back(
                llvm::json::Object{{"index", int64_t(position)},
                                   {"pack", int64_t(whole.getArgNumber())},
                                   {"offset", int64_t(0)},
                                   {"elements", wholeType.getDimSize(0)}});
            continue;
          }
          auto slice = operand.getDefiningOp<tensor::ExtractSliceOp>();
          auto argument = slice ? dyn_cast<BlockArgument>(slice.getSource())
                                : BlockArgument();
          if (argument && argument.getOwner() == &program.front() &&
              argument.getArgNumber() < packCount &&
              slice.getType().getRank() == 1 && slice.hasUnitStride() &&
              slice.getStaticOffsets()[0] != ShapedType::kDynamic &&
              slice.getStaticSizes()[0] != ShapedType::kDynamic) {
            described.push_back(
                llvm::json::Object{{"index", int64_t(position)},
                                   {"pack", int64_t(argument.getArgNumber())},
                                   {"offset", slice.getStaticOffsets()[0]},
                                   {"elements", slice.getStaticSizes()[0]}});
            continue;
          }
          record(builder, loc, 0, position, operand);
          described.push_back(llvm::json::Object{{"index", int64_t(position)}});
        }
        auto call = func::CallOp::create(builder, loc, callee, operands);
        for (auto [position, result] : llvm::enumerate(call.getResults()))
          record(builder, loc, 2, position, result);
        op->replaceAllUsesWith(call.getResults());
        auto hart = op->getAttrOfType<IntegerAttr>("hart");
        launches.push_back(
            llvm::json::Object{{"index", index},
                               {"program", name},
                               {"callee", callee.getName().str()},
                               {"hart", hart ? hart.getInt() : -1},
                               {"operands", std::move(described)}});
        op->erase();
        ++index;
      }

      FunctionType signature = program.getFunctionType();
      text << name << " " << signature.getNumInputs() << " " << packCount
           << "\n";
      llvm::interleave(
          signature.getResults(), text,
          [&](Type type) { text << cast<RankedTensorType>(type).getRank(); },
          " ");
      text << "\n";
      for (Type input : signature.getInputs()) {
        auto type = cast<RankedTensorType>(input);
        text << llvm::divideCeil(type.getElementTypeBitWidth(), 8) << " "
             << type.getRank();
        for (int64_t size : type.getShape())
          text << " " << (ShapedType::isDynamic(size) ? -1 : size);
        text << "\n";
      }
    }

    llvm::raw_fd_ostream file(layout, error);
    if (error)
      llvm::report_fatal_error("gvn-gpu-program: cannot write the layout");
    file << llvm::formatv("{0:2}", llvm::json::Value(llvm::json::Object{
                                       {"launches", std::move(launches)},
                                       {"values", std::move(numbers)},
                                       {"unrecorded", std::move(unrecorded)}}))
         << "\n";
  }
};
} // namespace

namespace gvn {
void registerGpuProgramPass() { PassRegistration<GpuProgramPass>(); }
} // namespace gvn

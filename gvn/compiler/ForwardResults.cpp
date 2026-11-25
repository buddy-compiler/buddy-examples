//===- ForwardResults.cpp - Pass the exposed values out of the entry ------===//
//
// gvn-forward-results{layout=<json>}, on the bufferized model entry (GPU
// side): the declared graph function it calls returns the values of the
// gvn-expose layout instead of its results, as identity-layout memrefs, and
// the entry returns them instead of its own results, which only return the
// results of the graph.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace mlir;

namespace {
class ForwardResultsPass
    : public PassWrapper<ForwardResultsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ForwardResultsPass)
  ForwardResultsPass() = default;
  ForwardResultsPass(const ForwardResultsPass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "gvn-forward-results"; }
  StringRef getDescription() const final {
    return "Return the exposed values of the graph from the model entry "
           "instead of its results";
  }

  Option<std::string> layout{*this, "layout",
                             llvm::cl::desc("The gvn-expose layout file")};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    auto buffer = llvm::MemoryBuffer::getFile(layout);
    if (!buffer)
      llvm::report_fatal_error("gvn-forward-results: cannot read the layout");
    auto parsed = llvm::json::parse((*buffer)->getBuffer());
    if (!parsed)
      llvm::report_fatal_error("gvn-forward-results: malformed layout");
    SmallVector<Type> exposed;
    for (const llvm::json::Value &value :
         *parsed->getAsObject()->getArray("values")) {
      const llvm::json::Object &entry = *value.getAsObject();
      SmallVector<int64_t> shape;
      for (const llvm::json::Value &size : *entry.getArray("shape"))
        shape.push_back(*size.getAsInteger());
      StringRef element = *entry.getString("element");
      Type type;
      if (element == "f32")
        type = Float32Type::get(context);
      else if (element == "f16")
        type = Float16Type::get(context);
      else if (element == "i8" || element == "i32" || element == "i64")
        type = IntegerType::get(context, element == "i8"    ? 8
                                         : element == "i32" ? 32
                                                            : 64);
      else
        llvm::report_fatal_error(
            "gvn-forward-results: unsupported element type");
      exposed.push_back(MemRefType::get(shape, type));
    }

    SmallVector<func::CallOp> calls;
    module.walk([&](func::CallOp call) {
      auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
      if (callee.isExternal())
        calls.push_back(call);
    });
    if (calls.size() != 1)
      llvm::report_fatal_error("gvn-forward-results: the entry must call one "
                               "declared graph function");
    func::CallOp call = calls.front();
    auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
    callee.setType(
        FunctionType::get(context, callee.getArgumentTypes(), exposed));
    auto entry = call->getParentOfType<func::FuncOp>();
    Operation *terminator = entry.front().getTerminator();
    for (Value result : call.getResults())
      for (Operation *user : result.getUsers())
        if (user != terminator)
          llvm::report_fatal_error("gvn-forward-results: the entry uses a "
                                   "result of the graph beyond returning it");

    OpBuilder builder(call);
    auto values = func::CallOp::create(builder, call.getLoc(), callee,
                                       call.getOperands());
    terminator->setOperands(values.getResults());
    call.erase();
    entry.setType(
        FunctionType::get(context, entry.getArgumentTypes(), exposed));
  }
};
} // namespace

namespace gvn {
void registerForwardResultsPass() { PassRegistration<ForwardResultsPass>(); }
} // namespace gvn

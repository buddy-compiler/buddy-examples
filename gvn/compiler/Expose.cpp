//===- Expose.cpp - Return the checked values of a linalg graph -----------===//
//
// gvn-expose{ids=<line:column,...> layout=<json>}, on the linalg graph the
// hart compiler started from (GPU side): the function returns, instead of
// its own results and in the order of `ids`, the result of each top-level
// operation located at `line:column`, the values the checks of the hart
// graph name. The layout file lists them with their shapes and elements.
// GPU code returns its results to the host; values written elsewhere stay in
// device memory.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
std::string elementName(Type type) {
  if (type.isF32())
    return "f32";
  if (type.isF16())
    return "f16";
  if (type.isInteger(8))
    return "i8";
  if (type.isInteger(32))
    return "i32";
  if (type.isInteger(64))
    return "i64";
  llvm::report_fatal_error("gvn: unsupported element type of a checked value");
}

class ExposePass : public PassWrapper<ExposePass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExposePass)
  ExposePass() = default;
  ExposePass(const ExposePass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "gvn-expose"; }
  StringRef getDescription() const final {
    return "Return the values the checks of the hart graph name instead of "
           "the results";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
  }

  ListOption<std::string> ids{*this, "ids",
                              llvm::cl::desc("line:column of each value")};
  Option<std::string> layout{*this, "layout",
                             llvm::cl::desc("The layout file to write")};

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (function.isExternal())
      return;
    Block &body = function.front();
    Operation *terminator = body.getTerminator();
    SmallVector<Value> results;
    llvm::json::Array values;
    for (const std::string &id : ids) {
      Value found;
      StringRef argumentId(id);
      if (argumentId.consume_front("arg:")) {
        unsigned argument;
        if (argumentId.getAsInteger(10, argument))
          llvm::report_fatal_error("gvn-expose: invalid argument index");
        found = function.getArgument(argument);
      }
      for (Operation &op : body.without_terminator()) {
        auto file = dyn_cast<FileLineColLoc>(op.getLoc());
        if (file && std::to_string(file.getLine()) + ":" +
                            std::to_string(file.getColumn()) ==
                        id) {
          if (op.getNumResults() != 1)
            llvm::report_fatal_error("gvn-expose: a checked operation has "
                                     "more than one result");
          found = op.getResult(0);
        }
      }
      if (!found)
        llvm::report_fatal_error(llvm::Twine("gvn-expose: no operation at ") +
                                 id);
      auto type = cast<RankedTensorType>(found.getType());
      if (isa<BlockArgument>(found) ||
          isa_and_nonnull<tensor::ExtractSliceOp, tensor::CollapseShapeOp,
                          tensor::ExpandShapeOp>(found.getDefiningOp())) {
        // Host-returned fixtures need their own contiguous allocation: a
        // reshape can view a strided slice (BERT's [CLS] row).
        // Materialize it before the GPU copy-lowering passes run.
        OpBuilder builder(terminator);
        found = bufferization::AllocTensorOp::create(builder, found.getLoc(),
                                                     type, ValueRange{}, found);
      }
      results.push_back(found);
      values.push_back(llvm::json::Object{
          {"id", id},
          {"shape", llvm::json::Array(type.getShape())},
          {"element", elementName(type.getElementType())},
          // A dynamic value's bytes are the GPU's to
          // tell: -1.
          {"bytes",
           type.hasStaticShape()
               ? type.getNumElements() * type.getElementTypeBitWidth() / 8
               : -1}});
    }
    terminator->setOperands(results);
    function.setType(FunctionType::get(function.getContext(),
                                       function.getArgumentTypes(),
                                       TypeRange(ValueRange(results))));
    std::error_code error;
    llvm::raw_fd_ostream file(layout, error);
    if (error)
      llvm::report_fatal_error("gvn-expose: cannot write the layout file");
    file << llvm::formatv("{0:2}", llvm::json::Value(llvm::json::Object{
                                       {"values", std::move(values)}}))
         << "\n";
  }
};
} // namespace

namespace gvn {
void registerExposePass() { PassRegistration<ExposePass>(); }
} // namespace gvn

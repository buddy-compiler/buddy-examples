//===- Import.cpp - Merge the modules of the device program ---------------===//
//
// gvn-merge, on a module holding modules (the device program and the linalg
// definitions of its graphs, each printed without aliases and concatenated):
// the functions of the inner modules join the outer one, a definition
// replacing a declaration of the same name, and the inner modules go. The
// modules are parsed together so that every dialect they use is loaded
// before the passes run.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
class MergePass : public PassWrapper<MergePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergePass)
  StringRef getArgument() const final { return "gvn-merge"; }
  StringRef getDescription() const final {
    return "Merge the functions of the inner modules into the outer one";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<ModuleOp> inner(module.getOps<ModuleOp>());
    for (ModuleOp other : inner) {
      for (auto function :
           llvm::make_early_inc_range(other.getOps<func::FuncOp>())) {
        auto existing = module.lookupSymbol<func::FuncOp>(function.getName());
        if (existing && (!existing.isExternal() || function.isExternal()))
          continue;
        if (existing)
          existing.erase();
        function->moveBefore(module.getBody(), module.getBody()->end());
      }
      other.erase();
    }
  }
};
} // namespace

namespace gvn {
void registerImportPass() { PassRegistration<MergePass>(); }
} // namespace gvn

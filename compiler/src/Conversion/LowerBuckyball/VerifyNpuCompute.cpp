#include "Conversion/LowerBuckyball/LowerBuckyball.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace {
class VerifyNpuComputePass
    : public PassWrapper<VerifyNpuComputePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyNpuComputePass)

  StringRef getArgument() const final { return "verify-npu-compute"; }
  StringRef getDescription() const final {
    return "Reject CPU floating computation in a final LLVM model kernel.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto isMath = [](StringRef name) {
      return llvm::StringSwitch<bool>(name)
          .Cases({"fadd", "fsub", "fmul", "fdiv", "frem", "fneg", "fcmp"}, true)
          .Cases({"pow", "powi", "exp", "exp2", "exp10", "expm1"}, true)
          .Cases({"log", "log2", "log10", "log1p", "sqrt", "cbrt"}, true)
          .Cases({"sin", "cos", "tan", "asin", "acos", "atan", "atan2"}, true)
          .Cases({"sinh", "cosh", "tanh", "asinh", "acosh", "atanh"}, true)
          .Cases({"erf", "erfc", "tgamma", "lgamma", "hypot"}, true)
          .Cases({"fabs", "copysign", "fma", "fmuladd", "fmod", "remainder"},
                 true)
          .Cases({"fmin", "fmax", "fdim", "minnum", "maxnum"}, true)
          .Cases({"minimum", "maximum", "minimumnum", "maximumnum"}, true)
          .Cases({"ceil", "floor", "trunc", "round", "roundeven"}, true)
          .Cases({"rint", "nearbyint", "lround", "llround", "lrint", "llrint"},
                 true)
          .Cases({"ldexp", "frexp", "scalbn", "scalbln", "ilogb", "logb"}, true)
          .Cases({"modf", "nextafter", "nexttoward", "sincos"}, true)
          .Default(false);
    };
    WalkResult result = getOperation().walk([&](Operation *op) {
      if (isa<ModuleOp>(op))
        return WalkResult::advance();
      if (op->getName().getDialectNamespace() != "llvm") {
        op->emitError("verify-npu-compute requires a final LLVM kernel");
        return WalkResult::interrupt();
      }
      StringRef name = op->getName().getStringRef();
      const bool arithmetic =
          isa<LLVM::FAddOp, LLVM::FSubOp, LLVM::FMulOp, LLVM::FDivOp,
              LLVM::FRemOp, LLVM::FNegOp, LLVM::FCmpOp>(op);
      bool call = false;
      if (auto direct = dyn_cast<LLVM::CallOp>(op)) {
        if (!direct.getCallee())
          return WalkResult::advance();
        name = *direct.getCallee();
        call = true;
      } else if (auto intrinsic = dyn_cast<LLVM::CallIntrinsicOp>(op)) {
        name = intrinsic.getIntrin();
        call = true;
      }
      bool intrinsic = name.consume_front("llvm.intr.");
      if (call || arithmetic)
        intrinsic |= name.consume_front("llvm.");
      if (intrinsic) {
        name.consume_front("experimental.constrained.");
        name = name.split('.').first;
      }
      bool rejected = arithmetic || ((call || intrinsic) && isMath(name));
      if (call && !intrinsic && !name.empty() &&
          (name.back() == 'f' || name.back() == 'l'))
        rejected |= isMath(name.drop_back());
      if (!rejected)
        return WalkResult::advance();
      op->emitError("CPU floating computation remains in NPU kernel: ") << name;
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace

void mlir::buddy::registerVerifyNpuComputePass() {
  PassRegistration<VerifyNpuComputePass>();
}

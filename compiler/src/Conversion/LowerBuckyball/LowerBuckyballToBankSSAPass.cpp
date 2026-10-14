//===- LowerBuckyballToBankSSAPass.cpp - Buckyball bank-SSA lowering
//----------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Buckyball/BuckyballDialect.h"
#include "Trace/TraceDialect.h"

using namespace mlir;

namespace mlir::buddy {
#define BUCKYBALL_BANK_SSA_HOOK(BALL)                                          \
  void populate##BALL##LowerBuckyballToBankSSAPatterns(RewritePatternSet &);
#include "BuckyballBallLoweringHooks.inc"
#undef BUCKYBALL_BANK_SSA_HOOK
} // namespace mlir::buddy

namespace {

class LowerBuckyballToBankSSAPass
    : public PassWrapper<LowerBuckyballToBankSSAPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBuckyballToBankSSAPass)

  StringRef getArgument() const final { return "lower-buckyball-to-bank-ssa"; }
  StringRef getDescription() const final {
    return "Lower Buckyball Buckyball ops to explicit bank-SSA ops.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<::buddy::trace::BuddyTraceDialect, arith::ArithDialect,
                    linalg::LinalgDialect, math::MathDialect,
                    memref::MemRefDialect, LLVM::LLVMDialect, scf::SCFDialect,
                    ::buddy::buckyball::BuckyballDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
#define BUCKYBALL_BANK_SSA_HOOK(BALL)                                          \
  mlir::buddy::populate##BALL##LowerBuckyballToBankSSAPatterns(patterns);
#include "BuckyballBallLoweringHooks.inc"
#undef BUCKYBALL_BANK_SSA_HOOK
    bool emittedError = false;
    ScopedDiagnosticHandler diagnostics(
        &getContext(), [&](Diagnostic &message) {
          emittedError |= message.getSeverity() == DiagnosticSeverity::Error;
          return failure();
        });
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))) ||
        emittedError)
      signalPassFailure();
  }
};

} // namespace

void mlir::buddy::registerLowerBuckyballToBankSSAPass() {
  PassRegistration<LowerBuckyballToBankSSAPass>();
}

//===- LowerBuckyball.cpp - Physical operations to Buckyball instructions -===//
//
// mset, mset_transfer, mvin and mvout become the instructions of bbhw/isa; each
// custom operation becomes the instructions the Ball of its kind emits. An
// instruction is a call of llvm.riscv.buckyball.custom with rs1, rs2 and the
// immediate funct7.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
constexpr int32_t kMvoutFunct7 = 16;
constexpr int32_t kMsetFunct7 = 32;
constexpr int32_t kMvinFunct7 = 33;

SmallVector<void (*)(RewritePatternSet &)> &ballLowerings() {
  static SmallVector<void (*)(RewritePatternSet &)> lowerings;
  return lowerings;
}

// Calls the runtime function `name` (i64 or i32 arguments, no result).
void callRuntime(OpBuilder &builder, Location loc, StringRef name,
                 ValueRange arguments) {
  auto module =
      builder.getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
  if (!module.lookupSymbol<func::FuncOp>(name)) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    func::FuncOp::create(builder, loc, name,
                         builder.getFunctionType(arguments.getTypes(), {}))
        .setPrivate();
  }
  func::CallOp::create(builder, loc, name, TypeRange{}, arguments);
}

Value constantI32(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIntOp::create(builder, loc, value, 32);
}

// The byte address of the first element of `memref`.
Value address(OpBuilder &builder, Location loc, Value memref) {
  auto type = cast<MemRefType>(memref.getType());
  Value base =
      memref::ExtractAlignedPointerAsIndexOp::create(builder, loc, memref);
  Value offset = memref::ExtractStridedMetadataOp::create(builder, loc, memref)
                     .getOffset();
  Value bytes = arith::MulIOp::create(
      builder, loc, offset,
      arith::ConstantIndexOp::create(builder, loc,
                                     type.getElementTypeBitWidth() / 8));
  Value element = arith::AddIOp::create(builder, loc, base, bytes);
  return arith::IndexCastOp::create(builder, loc, builder.getI64Type(),
                                    element);
}

// rs2 of mvin and mvout: the memory address and the stride in 16-byte rows.
Value memoryOperand(OpBuilder &builder, Location loc, Value memoryAddress,
                    int64_t stride) {
  Value addressField = arith::AndIOp::create(
      builder, loc, memoryAddress,
      mlir::buddy::constantI64(builder, loc, (1ULL << 39) - 1));
  return arith::OrIOp::create(
      builder, loc, addressField,
      mlir::buddy::constantI64(builder, loc,
                               mlir::buddy::field(stride, 39, 57)));
}

class MsetLowering : public OpRewritePattern<MsetOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MsetOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t bank = op.getBank();
    // The Linux DMA service tracks the columns of every bank.
    callRuntime(rewriter, loc,
                op.getAlloc() ? "dma_bank_allocate" : "dma_bank_set_cols",
                {constantI32(rewriter, loc, bank),
                 constantI32(rewriter, loc, op.getCol())});
    uint64_t rs2 = mlir::buddy::field(op.getRow(), 0, 4) |
                   mlir::buddy::field(op.getCol(), 5, 9) |
                   mlir::buddy::field(op.getAlloc(), 10, 10);
    mlir::buddy::emitInstruction(
        rewriter, loc, kMsetFunct7,
        mlir::buddy::constantI64(rewriter, loc, mlir::buddy::field(bank, 0, 9)),
        mlir::buddy::constantI64(rewriter, loc, rs2));
    rewriter.eraseOp(op);
    return success();
  }
};

class MsetTransferLowering : public OpRewritePattern<MsetTransferOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MsetTransferOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // The Linux DMA service moves the columns it tracks for the banks.
    callRuntime(rewriter, loc, "dma_bank_transfer",
                {mlir::buddy::constantI64(rewriter, loc, op.getSource()),
                 mlir::buddy::constantI64(rewriter, loc, op.getTarget())});
    uint64_t rs1 = mlir::buddy::field(op.getSource(), 0, 9) |
                   mlir::buddy::field(op.getTarget(), 20, 28);
    mlir::buddy::emitInstruction(
        rewriter, loc, kMsetFunct7,
        mlir::buddy::constantI64(rewriter, loc, rs1),
        mlir::buddy::constantI64(rewriter, loc, mlir::buddy::field(1, 12, 12)));
    rewriter.eraseOp(op);
    return success();
  }
};

class MvinLowering : public OpRewritePattern<MvinOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MvinOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    uint64_t rs1 =
        mlir::buddy::bankFields({}, op.getBank(), false, op.getDepth());
    mlir::buddy::emitInstruction(
        rewriter, loc, kMvinFunct7,
        mlir::buddy::constantI64(rewriter, loc, rs1),
        memoryOperand(rewriter, loc, address(rewriter, loc, op.getInput()),
                      op.getStride()));
    rewriter.eraseOp(op);
    return success();
  }
};

class MvoutLowering : public OpRewritePattern<MvoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MvoutOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value memoryAddress = address(rewriter, loc, op.getOutput());
    // Linux needs private destination pages before a DMA store.
    callRuntime(rewriter, loc, "dma_touch_mvout",
                {memoryAddress,
                 mlir::buddy::constantI64(rewriter, loc, op.getDepth()),
                 mlir::buddy::constantI64(rewriter, loc, op.getStride()),
                 constantI32(rewriter, loc, op.getBank())});
    uint64_t rs1 = mlir::buddy::bankFields({static_cast<int64_t>(op.getBank())},
                                           0, false, op.getDepth());
    mlir::buddy::emitInstruction(
        rewriter, loc, kMvoutFunct7,
        mlir::buddy::constantI64(rewriter, loc, rs1),
        memoryOperand(rewriter, loc, memoryAddress, op.getStride()));
    rewriter.eraseOp(op);
    return success();
  }
};

class LowerBuckyballPass
    : public PassWrapper<LowerBuckyballPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBuckyballPass)

  StringRef getArgument() const final { return "lower-buckyball"; }
  StringRef getDescription() const final {
    return "Lower the physical Buckyball operations to instructions";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns
        .add<MsetLowering, MsetTransferLowering, MvinLowering, MvoutLowering>(
            &getContext());
    for (auto populate : ballLowerings())
      populate(patterns);
    // A Ball that rejects an operation reports why; it is not reported again
    // as an operation without a lowering.
    bool rejected = false;
    ScopedDiagnosticHandler handler(&getContext(), [&](Diagnostic &diagnostic) {
      rejected |= diagnostic.getSeverity() == DiagnosticSeverity::Error;
      return failure();
    });
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))) ||
        rejected)
      return signalPassFailure();
    WalkResult left = getOperation().walk([](Operation *op) {
      if (isa<BuckyballDialect>(op->getDialect())) {
        op->emitOpError("has no lowering on this core");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (left.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace

namespace mlir::buddy {
void registerAssignPhysicalBanksPass();
void registerLowerFrontendPass();

void registerBuckyballPasses() {
  registerAssignPhysicalBanksPass();
  registerLowerFrontendPass();
  PassRegistration<LowerBuckyballPass>();
}

void addBallLowering(void (*populate)(RewritePatternSet &patterns)) {
  ballLowerings().push_back(populate);
}

void emitInstruction(OpBuilder &builder, Location loc, int32_t funct7,
                     Value rs1, Value rs2) {
  Value immediate = LLVM::ConstantOp::create(builder, loc, builder.getI32Type(),
                                             builder.getI32IntegerAttr(funct7));
  LLVM::CallIntrinsicOp::create(
      builder, loc, builder.getStringAttr("llvm.riscv.buckyball.custom"),
      ValueRange{rs1, rs2, immediate});
}

Value constantI64(OpBuilder &builder, Location loc, uint64_t value) {
  return arith::ConstantIntOp::create(builder, loc, static_cast<int64_t>(value),
                                      64);
}

uint64_t field(uint64_t value, unsigned start, unsigned end) {
  return (value & ((2ULL << (end - start)) - 1)) << start;
}

uint64_t bankFields(ArrayRef<int64_t> sources, int64_t destination,
                    bool readDestination, uint64_t iterations) {
  uint64_t rs1 = field(destination, 20, 28) | field(readDestination, 29, 29) |
                 field(iterations, 30, 62);
  if (!sources.empty())
    rs1 |= field(sources[0], 0, 9);
  if (sources.size() == 2)
    rs1 |= field(sources[1], 10, 19);
  return rs1;
}
} // namespace mlir::buddy

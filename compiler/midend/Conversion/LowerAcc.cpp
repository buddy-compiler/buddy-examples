//===- LowerAcc.cpp - Physical operations to Buckyball instructions -------===//
//
// mset, mset_transfer, mvin, mvout and the GP domain's mvin_group,
// mvin_operator and run_operator become base instructions; each Ball operation
// emits its own instructions through PhysicalBankAssignment. An instruction
// is a call of llvm.riscv.buckyball.custom with rs1, rs2 and the immediate
// funct7.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccDialect.h"
#include "Acc/AccOps.h"
#include "Acc/BankInterfaces.h"
#include "Conversion/Buckyball.h"
#include "Target/TargetRegistry.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace ::buddy::acc;

namespace {
constexpr int32_t kMvoutFunct7 = 16;
constexpr int32_t kMsetFunct7 = 32;
constexpr int32_t kMvinFunct7 = 33;
constexpr int32_t kMvinOperatorFunct7 = 44;
constexpr int32_t kRunOperatorFunct7 = 79;

// The byte address of the first element of `memref`.
Value address(OpBuilder &builder, Location loc, Value memref) {
  auto type = cast<MemRefType>(memref.getType());
  // These views change descriptor offsets and strides, not the aligned
  // allocation pointer. Keep loop offsets out of the pointer expression.
  Value storage = memref;
  while (Operation *view = storage.getDefiningOp()) {
    if (!isa<memref::CastOp, memref::SubViewOp, memref::ReinterpretCastOp,
             memref::CollapseShapeOp, memref::ExpandShapeOp>(view))
      break;
    storage = view->getOperand(0);
  }
  Value base =
      memref::ExtractAlignedPointerAsIndexOp::create(builder, loc, storage);
  Value offset = memref::ExtractStridedMetadataOp::create(builder, loc, memref)
                     .getOffset();
  Value bytes = arith::MulIOp::create(
      builder, loc, offset,
      arith::ConstantIndexOp::create(builder, loc,
                                     (type.getElementTypeBitWidth() + 7) / 8));
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

class ClearLowering : public OpRewritePattern<ClearOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ClearOp op, PatternRewriter &b) const override {
    uint64_t rs2 = mlir::buddy::field(op.getRow(), 0, 4) |
                   mlir::buddy::field(op.getCol(), 5, 9) | (1 << 10) |
                   (1 << 11);
    mlir::buddy::emitInstruction(
        b, op.getLoc(), kMsetFunct7,
        mlir::buddy::constantI64(b, op.getLoc(), op.getBank()),
        mlir::buddy::constantI64(b, op.getLoc(), rs2));
    b.eraseOp(op);
    return success();
  }
};

class MsetTransferLowering : public OpRewritePattern<MsetTransferOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MsetTransferOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
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

class MvinGroupLowering : public OpRewritePattern<MvinGroupOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MvinGroupOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    uint64_t rs1 =
        mlir::buddy::bankFields({}, op.getBank(), false, op.getDepth());
    Value rs2 = arith::OrIOp::create(
        rewriter, loc,
        memoryOperand(rewriter, loc, address(rewriter, loc, op.getInput()),
                      op.getStride()),
        mlir::buddy::constantI64(rewriter, loc,
                                 mlir::buddy::field(op.getGroup(), 58, 62) |
                                     mlir::buddy::field(1, 63, 63)));
    mlir::buddy::emitInstruction(rewriter, loc, kMvinFunct7,
                                 mlir::buddy::constantI64(rewriter, loc, rs1),
                                 rs2);
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename Op, bool Write>
class StridedDmaLowering : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &b) const override {
    Location loc = op.getLoc();
    if (op.getGroup() < 0 || op.getGroup() >= 32 || op.getBankStride() < 1 ||
        op.getBankStride() > (Write ? 512 : 1024))
      return op.emitOpError("strided DMA fields exceed the ISA");
    if (auto first = getConstantIntValue(op.getFirst());
        first && (*first < 0 || *first > 1023))
      return op.emitOpError("strided DMA first bank row exceeds ten bits");
    auto shift = [&](Value value, int64_t bits) -> Value {
      return arith::ShLIOp::create(
          b, loc, arith::IndexCastOp::create(b, loc, b.getI64Type(), value),
          mlir::buddy::constantI64(b, loc, bits));
    };
    uint64_t fields =
        Write ? mlir::buddy::field(op.getBank(), 0, 9) |
                    mlir::buddy::field(op.getBankStride() - 1, 20, 28)
              : mlir::buddy::field(op.getBankStride() - 1, 10, 19) |
                    mlir::buddy::field(op.getBank(), 20, 28) |
                    (UINT64_C(1) << 63);
    // These fields are disjoint. Addition preserves their affine loop
    // increments for the instruction-stream lowering.
    Value rs1 = arith::AddIOp::create(
        b, loc, mlir::buddy::constantI64(b, loc, fields),
        arith::AddIOp::create(b, loc, shift(op.getFirst(), Write ? 10 : 0),
                              shift(op.getDepth(), 30)));
    // Keep the address increment affine when byte count or stride is a
    // runtime value. Address and descriptor fields occupy disjoint bits.
    Value pointer = arith::AndIOp::create(
        b, loc, address(b, loc, op.getMemory()),
        mlir::buddy::constantI64(b, loc, (UINT64_C(1) << 39) - 1));
    // Whole rows a DDR stride of 4096 bytes or more apart exceed the strided
    // DMA's byte stride: MVIN_BLOCKS / MVOUT_BLOCKS count the stride in rows.
    auto bytes = getConstantIntValue(op.getBytes());
    auto stride = getConstantIntValue(op.getMemoryStride());
    if (bytes && stride && *bytes == 16 && *stride >= 4096) {
      if (*stride % 16 || *stride / 16 >= 65536)
        return op.emitOpError("block DMA stride must be whole rows below 1 MB");
      Value rs2 = arith::AddIOp::create(
          b, loc, pointer,
          mlir::buddy::constantI64(
              b, loc,
              mlir::buddy::field(*stride / 16, 39, 54) |
                  mlir::buddy::field(op.getGroup(), 55, 59)));
      rs2 = arith::AddIOp::create(b, loc, rs2, shift(op.getBlockLog2(), 60));
      mlir::buddy::emitInstruction(b, loc, Write ? 19 : 37, rs1, rs2);
      b.eraseOp(op);
      return success();
    }
    Value rs2 = arith::AddIOp::create(
        b, loc, pointer,
        arith::OrIOp::create(
            b, loc, shift(op.getMemoryStride(), 39),
            arith::OrIOp::create(
                b, loc,
                shift(arith::SubIOp::create(
                          b, loc, op.getBytes(),
                          arith::ConstantIndexOp::create(b, loc, 1)),
                      51),
                mlir::buddy::constantI64(
                    b, loc, mlir::buddy::field(op.getGroup(), 55, 59)))));
    rs2 = arith::AddIOp::create(b, loc, rs2, shift(op.getBlockLog2(), 60));
    mlir::buddy::emitInstruction(b, loc, Write ? 18 : 36, rs1, rs2);
    b.eraseOp(op);
    return success();
  }
};

class MvinOperatorLowering : public OpRewritePattern<MvinOperatorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MvinOperatorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (!target::getTarget().rvvEnabled)
      return op.emitOpError("needs a GP domain on this target");
    Value rs1 = arith::OrIOp::create(
        rewriter, loc,
        arith::ShLIOp::create(rewriter, loc, op.getBytes(),
                              mlir::buddy::constantI64(rewriter, loc, 30)),
        mlir::buddy::constantI64(rewriter, loc,
                                 mlir::buddy::field(op.getProgram(), 20, 28)));
    mlir::buddy::emitInstruction(rewriter, loc, kMvinOperatorFunct7, rs1,
                                 address(rewriter, loc, op.getImage()));
    rewriter.eraseOp(op);
    return success();
  }
};

class RunOperatorLowering : public OpRewritePattern<RunOperatorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RunOperatorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    uint64_t rs1 =
        mlir::buddy::bankFields({static_cast<int64_t>(op.getInput()),
                                 static_cast<int64_t>(op.getProgram())},
                                op.getOutput(), false, 0);
    mlir::buddy::emitInstruction(rewriter, loc, kRunOperatorFunct7,
                                 mlir::buddy::constantI64(rewriter, loc, rs1),
                                 mlir::buddy::constantI64(rewriter, loc, 0));
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

class LowerAccPass : public PassWrapper<LowerAccPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAccPass)

  StringRef getArgument() const final { return "lower-acc"; }
  StringRef getDescription() const final {
    return "Lower the physical acc operations to Buckyball instructions";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, LLVM::LLVMDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    SmallVector<func::FuncOp> functions(getOperation().getOps<func::FuncOp>());
    for (auto function : functions) {
      if (auto profile = function->getAttrOfType<StringAttr>("bbacc.target");
          profile && profile.getValue() != target::getTarget().hart)
        continue;
      SmallVector<PhysicalBankAssignment> balls;
      function.walk([&](PhysicalBankAssignment op) { balls.push_back(op); });
      for (PhysicalBankAssignment op : balls) {
        OpBuilder builder(op);
        if (failed(op.lowerToInstructions(builder)))
          return signalPassFailure();
        op->erase();
      }
      RewritePatternSet patterns(&getContext());
      patterns.add<ClearLowering, StridedDmaLowering<MvinStridedOp, false>,
                   StridedDmaLowering<MvoutStridedOp, true>>(&getContext());
      patterns
          .add<MsetLowering, MsetTransferLowering, MvinLowering, MvoutLowering,
               MvinGroupLowering, MvinOperatorLowering, RunOperatorLowering>(
              &getContext());
      if (failed(applyPatternsGreedily(function, std::move(patterns))))
        return signalPassFailure();
      WalkResult left = function.walk([](Operation *op) {
        if (isa<AccDialect>(op->getDialect()) || isa<BankSSALowering>(op)) {
          op->emitOpError("has no lowering on this hart");
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (left.wasInterrupted())
        signalPassFailure();
    }
  }
};
} // namespace

namespace mlir::buddy {
void registerAssignPhysicalBanksPass();
void registerLowerFrontendFakeOpPass();
void registerMaterializeAccResultsPass();

void registerBuckyballPasses() {
  registerAssignPhysicalBanksPass();
  registerLowerFrontendFakeOpPass();
  registerMaterializeAccResultsPass();
  PassRegistration<LowerAccPass>();
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

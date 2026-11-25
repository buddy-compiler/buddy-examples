//===- Stream.cpp - Record thread instructions and affine operand loops --===//

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

using namespace mlir;

namespace {
// An instruction operand's increment for one iteration. The lowering packs
// disjoint fields; shifts retain the affine increment within each field.
FailureOr<Value> increment(OpBuilder &b, Location loc, Value value, Value iv) {
  auto constant = [&](int64_t n) -> Value {
    return arith::ConstantIntOp::create(b, loc, n, 64);
  };
  if (value == iv)
    return constant(1);
  Operation *op = value.getDefiningOp();
  if (!op)
    return constant(0);
  SmallVector<Value> terms;
  for (Value operand : op->getOperands()) {
    auto term = increment(b, loc, operand, iv);
    if (failed(term))
      return failure();
    terms.push_back(*term);
  }
  if (llvm::all_of(terms,
                   [](Value term) { return matchPattern(term, m_Zero()); }))
    return constant(0);
  if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(op))
    return terms[0];
  if (isa<arith::AddIOp>(op))
    return b.createOrFold<arith::AddIOp>(loc, terms[0], terms[1]);
  if (isa<arith::SubIOp>(op))
    return b.createOrFold<arith::SubIOp>(loc, terms[0], terms[1]);
  if (isa<arith::ShLIOp>(op) && matchPattern(terms[1], m_Zero()))
    return b.createOrFold<arith::ShLIOp>(loc, terms[0], op->getOperand(1));
  if (isa<arith::MulIOp>(op)) {
    for (unsigned side = 0; side < 2; ++side)
      if (matchPattern(terms[side], m_Zero())) {
        Value scale = op->getOperand(side);
        if (scale.getType().isIndex())
          scale =
              b.createOrFold<arith::IndexCastOp>(loc, b.getI64Type(), scale);
        return b.createOrFold<arith::MulIOp>(loc, scale, terms[1 - side]);
      }
  }
  APInt mask;
  if (op->getNumOperands() == 2 &&
      matchPattern(op->getOperand(1), m_ConstantInt(&mask))) {
    uint64_t bits = mask.getZExtValue();
    if (isa<arith::AndIOp>(op) && bits == (1ULL << 39) - 1)
      return terms[0];
    if (isa<arith::OrIOp>(op) && !(bits & ((1ULL << 39) - 1)))
      return terms[0];
  }
  return failure();
}
} // namespace

namespace mlir::buddy {
LogicalResult lowerThreadStream(func::FuncOp function) {
  ModuleOp module = function->getParentOfType<ModuleOp>();
  OpBuilder b(function.getContext());
  Type i64 = b.getI64Type(), i32 = b.getI32Type();
  Type ptr = LLVM::LLVMPointerType::get(b.getContext());
  auto declare = [&](StringRef name, TypeRange types) {
    if (auto existing = module.lookupSymbol<func::FuncOp>(name))
      return existing;
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(module.getBody());
    auto callee = func::FuncOp::create(b, function.getLoc(), name,
                                       b.getFunctionType(types, {}));
    callee.setPrivate();
    return callee;
  };
  auto emit = declare("stream_emit", {i32, i64, i64});
  auto offset = declare("stream_offset", {i64, i64, i64});
  auto begin = declare("stream_loop", {i64});
  auto end = declare("stream_end", {});
  auto release = declare("stream_release", {ptr});
  auto allocate = declare("stream_allocate", {i64, i64});
  allocate.setFunctionType(b.getFunctionType({i64, i64}, {ptr}));
  SmallVector<memref::AllocOp> allocations;
  SmallVector<LLVM::CallIntrinsicOp> instructions;
  SmallVector<memref::DeallocOp> deallocations;
  SmallVector<memref::CopyOp> copies;
  SmallVector<scf::ForOp> loops;
  function.walk([&](Operation *op) {
    if (auto allocation = dyn_cast<memref::AllocOp>(op))
      allocations.push_back(allocation);
    if (auto intrinsic = dyn_cast<LLVM::CallIntrinsicOp>(op))
      if (intrinsic.getIntrin() == "llvm.riscv.buckyball.custom")
        instructions.push_back(intrinsic);
    if (auto copy = dyn_cast<memref::CopyOp>(op);
        copy && copy->hasAttr("hart.after"))
      copies.push_back(copy);
    if (auto dealloc = dyn_cast<memref::DeallocOp>(op))
      deallocations.push_back(dealloc);
    if (auto loop = dyn_cast<scf::ForOp>(op);
        loop && loop->hasAttr("hart.stream"))
      loops.push_back(loop);
  });
  // Recording evaluates host packing immediately. Its input buffers must
  // already be ready, even though the recorded DMA commands run later.
  SmallVector<int64_t> hostInputs, hostWrites;
  function.walk([&](Operation *op) {
    auto effects = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effects || op->hasAttr("hart.after"))
      return;
    for (Value value : op->getOperands()) {
      if (!isa<MemRefType>(value.getType()))
        continue;
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffectsOnValue(value, instances);
      bool reads = llvm::any_of(instances, [](const auto &effect) {
        return isa<MemoryEffects::Read>(effect.getEffect());
      });
      bool writes = llvm::any_of(instances, [](const auto &effect) {
        return isa<MemoryEffects::Write>(effect.getEffect());
      });
      if (!reads && !writes)
        continue;
      while (auto view =
                 dyn_cast_or_null<ViewLikeOpInterface>(value.getDefiningOp()))
        value = view.getViewSource();
      if (auto argument = dyn_cast<BlockArgument>(value);
          argument && argument.getOwner() == &function.getBody().front()) {
        if (reads)
          hostInputs.push_back(argument.getArgNumber());
        if (writes)
          hostWrites.push_back(argument.getArgNumber());
      }
    }
  });
  llvm::sort(hostInputs);
  hostInputs.erase(std::unique(hostInputs.begin(), hostInputs.end()),
                   hostInputs.end());
  function->setAttr("hart.host_inputs", b.getDenseI64ArrayAttr(hostInputs));
  llvm::sort(hostWrites);
  hostWrites.erase(std::unique(hostWrites.begin(), hostWrites.end()),
                   hostWrites.end());
  function->setAttr("hart.host_writes", b.getDenseI64ArrayAttr(hostWrites));
  if (!copies.empty())
    function->setAttr("hart.host_completion", b.getUnitAttr());
  for (auto instruction : instructions) {
    b.setInsertionPoint(instruction);
    Location loc = instruction.getLoc();
    SmallVector<Value> indices;
    for (Operation *parent = instruction->getParentOp(); parent != function;
         parent = parent->getParentOp()) {
      if (auto loop = dyn_cast<scf::ForOp>(parent)) {
        // Explicit recording loops run on the host and emit each chosen
        // stream, including its constant instruction fields and tail branch.
        if (loop->hasAttr("hart.record"))
          continue;
        if (!loop->hasAttr("hart.stream"))
          return loop.emitError(
              "Ball instructions need a recorded stream loop");
        indices.push_back(loop.getInductionVar());
      }
    }
    if (indices.size() > 8)
      return instruction.emitError(
          "instruction stream exceeds eight nested loops");
    for (auto [depth, iv] : llvm::enumerate(llvm::reverse(indices))) {
      auto rs1 = increment(b, loc, instruction.getArgs()[0], iv);
      auto rs2 = increment(b, loc, instruction.getArgs()[1], iv);
      if (failed(rs1) || failed(rs2))
        return instruction.emitError(
            "stream loop requires affine instruction operands");
      for (auto [operand, delta] : llvm::enumerate(ArrayRef<Value>{*rs1, *rs2}))
        if (!matchPattern(delta, m_Zero()))
          func::CallOp::create(
              b, loc, offset,
              ValueRange{arith::ConstantIntOp::create(b, loc, depth, 64),
                         arith::ConstantIntOp::create(b, loc, operand, 64),
                         delta});
    }
    func::CallOp::create(b, loc, emit,
                         ValueRange{instruction.getArgs()[2],
                                    instruction.getArgs()[0],
                                    instruction.getArgs()[1]});
    instruction.erase();
  }
  // walk is post-order: flatten the inner loops first. Scalar address
  // computations run once on the hart; the thread adds iteration offsets.
  for (auto loop : loops) {
    APInt lower, step;
    if (loop.getNumResults() ||
        !matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
        !lower.isZero() ||
        !matchPattern(loop.getStep(), m_ConstantInt(&step)) || step != 1)
      return loop.emitError("stream loop needs zero lower bound, unit step and "
                            "no carried values");
    b.setInsertionPoint(loop);
    Location loc = loop.getLoc();
    // Empty tail tiles need neither recorded instructions nor DMA preparation.
    Value nonempty = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::ugt, loop.getUpperBound(),
        arith::ConstantIndexOp::create(b, loc, 0));
    auto record = scf::IfOp::create(b, loc, nonempty, false);
    b.setInsertionPointToStart(&record.getThenRegion().front());
    Value count = arith::IndexCastOp::create(b, loc, i64, loop.getUpperBound());
    func::CallOp::create(b, loc, begin, ValueRange{count});
    loop.getInductionVar().replaceAllUsesWith(
        arith::ConstantIndexOp::create(b, loc, 0));
    loop.getBody()->getTerminator()->erase();
    record.getThenRegion().front().getOperations().splice(
        b.getInsertionPoint(), loop.getBody()->getOperations());
    func::CallOp::create(b, loc, end, ValueRange{});
    loop.erase();
  }
  auto copyRows = declare("stream_copy", {ptr, ptr, i64, i64, i64, i64});
  for (auto copy : copies) {
    b.setInsertionPoint(copy);
    Location loc = copy.getLoc();
    auto type = copy.getSource().getType();
    Value bytesPerElement = arith::ConstantIndexOp::create(
        b, loc, type.getElementTypeBitWidth() / 8);
    Value width = b.createOrFold<arith::MulIOp>(
        loc,
        memref::DimOp::create(b, loc, copy.getSource(), type.getRank() - 1)
            .getResult(),
        bytesPerElement);
    Value rows = arith::ConstantIndexOp::create(b, loc, 1);
    for (int64_t axis = 0; axis + 1 < type.getRank(); ++axis)
      rows = b.createOrFold<arith::MulIOp>(
          loc, rows,
          memref::DimOp::create(b, loc, copy.getSource(), axis).getResult());
    SmallVector<Value> pointers, strides;
    for (Value memory : {copy.getTarget(), copy.getSource()}) {
      auto layout = memref::ExtractStridedMetadataOp::create(b, loc, memory);
      Value base =
          memref::ExtractAlignedPointerAsIndexOp::create(b, loc, memory);
      Value bytes = arith::ConstantIndexOp::create(
          b, loc, type.getElementTypeBitWidth() / 8);
      Value address = arith::AddIOp::create(
          b, loc, base,
          arith::MulIOp::create(b, loc, layout.getOffset(), bytes));
      pointers.push_back(LLVM::IntToPtrOp::create(
          b, loc, ptr,
          arith::IndexCastOp::create(b, loc, i64, address).getResult()));
      Value stride =
          type.getRank() > 1
              ? layout.getStrides()[type.getRank() - 2]
              : arith::ConstantIndexOp::create(b, loc, type.getShape().back())
                    .getResult();
      strides.push_back(arith::IndexCastOp::create(
          b, loc, i64, arith::MulIOp::create(b, loc, stride, bytes)));
    }
    func::CallOp::create(
        b, loc, copyRows,
        ValueRange{pointers[0], pointers[1],
                   arith::IndexCastOp::create(b, loc, i64, rows),
                   arith::IndexCastOp::create(b, loc, i64, width), strides[0],
                   strides[1]});
    copy.erase();
  }
  LLVMTypeConverter converter(b.getContext());
  // The recorded job returns before its DMA runs. Keep its staging buffers
  // with the batch, including their lifetime metadata in the same allocation.
  llvm::erase_if(deallocations, [&](memref::DeallocOp dealloc) {
    Value source = dealloc.getMemref();
    while (auto view =
               dyn_cast_or_null<ViewLikeOpInterface>(source.getDefiningOp()))
      source = view.getViewSource();
    if (!source.getDefiningOp<memref::AllocOp>())
      return false;
    dealloc.erase();
    return true;
  });
  for (auto allocation : allocations) {
    b.setInsertionPoint(allocation);
    Location loc = allocation.getLoc();
    MemRefType type = allocation.getType();
    if (!type.getLayout().isIdentity() || type.getMemorySpace())
      return allocation.emitError(
          "stream staging needs an identity DDR buffer");
    SmallVector<Value> sizes;
    unsigned dynamic = 0;
    Value elements = arith::ConstantIntOp::create(b, loc, 1, 64);
    for (int64_t dim : type.getShape()) {
      Value size =
          ShapedType::isDynamic(dim)
              ? arith::IndexCastOp::create(
                    b, loc, i64, allocation.getDynamicSizes()[dynamic++])
                    .getResult()
              : arith::ConstantIntOp::create(b, loc, dim, 64).getResult();
      sizes.push_back(size);
      elements = arith::MulIOp::create(b, loc, elements, size);
    }
    Value bytes = arith::MulIOp::create(
        b, loc, elements,
        arith::ConstantIntOp::create(
            b, loc, (type.getElementTypeBitWidth() + 7) / 8, 64));
    Value alignment = arith::ConstantIntOp::create(
        b, loc, allocation.getAlignment().value_or(8), 64);
    Value address =
        func::CallOp::create(b, loc, allocate, ValueRange{bytes, alignment})
            .getResult(0);
    MemRefDescriptor descriptor =
        MemRefDescriptor::poison(b, loc, converter.convertType(type));
    descriptor.setAllocatedPtr(b, loc, address);
    descriptor.setAlignedPtr(b, loc, address);
    descriptor.setConstantOffset(b, loc, 0);
    Value stride = arith::ConstantIntOp::create(b, loc, 1, 64);
    for (int64_t axis = type.getRank() - 1; axis >= 0; --axis) {
      descriptor.setSize(b, loc, axis, sizes[axis]);
      descriptor.setStride(b, loc, axis, stride);
      stride = arith::MulIOp::create(b, loc, stride, sizes[axis]);
    }
    Value buffer =
        UnrealizedConversionCastOp::create(b, loc, type, Value(descriptor))
            .getResult(0);
    allocation.replaceAllUsesWith(buffer);
    allocation.erase();
  }
  for (auto dealloc : deallocations) {
    b.setInsertionPoint(dealloc);
    Location loc = dealloc.getLoc();
    Value descriptor =
        UnrealizedConversionCastOp::create(
            b, loc, converter.convertType(dealloc.getMemref().getType()),
            dealloc.getMemref())
            .getResult(0);
    Value pointer = MemRefDescriptor(descriptor).allocatedPtr(b, loc);
    func::CallOp::create(b, loc, release, ValueRange{pointer});
    dealloc.erase();
  }
  // Reserve consecutive recorded instructions once, then store their fields
  // directly. Loop delimiters retain their runtime bookkeeping.
  SmallVector<SmallVector<func::CallOp>> groups;
  function.walk([&](Block *block) {
    SmallVector<func::CallOp> group;
    for (Operation &op : *block) {
      auto call = dyn_cast<func::CallOp>(op);
      if (call && (call.getCallee() == "stream_emit" ||
                   call.getCallee() == "stream_offset")) {
        group.push_back(call);
      } else if (call || op.getNumRegions()) {
        if (group.size() > 1)
          groups.push_back(group);
        group.clear();
      }
    }
    if (group.size() > 1)
      groups.push_back(group);
  });
  if (!groups.empty()) {
    auto reserve = module.lookupSymbol<func::FuncOp>("stream_reserve");
    if (!reserve) {
      b.setInsertionPointToEnd(module.getBody());
      reserve = func::FuncOp::create(b, function.getLoc(), "stream_reserve",
                                     b.getFunctionType({i64}, {ptr}));
      reserve.setPrivate();
    }
    for (auto group : groups) {
      b.setInsertionPoint(group.front());
      Location loc = group.front().getLoc();
      Value target =
          func::CallOp::create(b, loc, reserve,
                               ValueRange{arith::ConstantIntOp::create(
                                   b, loc, group.size(), 64)})
              .getResult(0);
      for (auto [index, call] : llvm::enumerate(group)) {
        b.setInsertionPoint(call);
        loc = call.getLoc();
        SmallVector<Value> fields;
        if (call.getCallee() == "stream_emit")
          fields = {
              b.createOrFold<arith::ExtUIOp>(loc, i64, call.getOperand(0)),
              call.getOperand(1), call.getOperand(2)};
        else
          fields = {b.createOrFold<arith::AddIOp>(
                        loc, call.getOperand(1),
                        arith::ConstantIntOp::create(b, loc, 130, 64)),
                    call.getOperand(0), call.getOperand(2)};
        for (auto [field, value] : llvm::enumerate(fields)) {
          Value at = LLVM::GEPOp::create(
              b, loc, ptr, i64, target,
              ArrayRef<LLVM::GEPArg>{int32_t(index * 3 + field)});
          LLVM::StoreOp::create(b, loc, value, at);
        }
        call.erase();
      }
    }
  }
  return success();
}
} // namespace mlir::buddy

//===- LinalgToGPUKernels.cpp - Thread-block kernels for linalg ops -------===//
//
// The NVIDIA backend's own pass (backends share no code).
//
// Every linalg.generic on buffers (run linalg-generalize-named-ops first)
// becomes one gpu.launch:
//
//   - Without reduction loops: one thread per iteration, 256 threads per
//     block over the linearized iteration space (innermost dimension fastest).
//
//   - With reduction loops: a thread-block tile of the output. The parallel
//     dimensions split into three groups by the inputs that read them:
//     N, the dimensions read by the inputs that read the output's innermost
//     dimension (the B inputs) and by no other input; M, the dimensions read
//     only by the other inputs (the A inputs); the rest are batch dimensions.
//     M, N, the batch and the reduction dimensions are each linearized. A
//     block computes BM x BN outputs of one batch index with TX x TY
//     threads, each thread TM x TN outputs held in registers
//     (row ty + i * TY, column tx + j * TX). The reduction runs in chunks of
//     BK: the block stages the chunk's A values (BK x BM) and B values
//     (BK x BN) in workgroup memory (gpu.barrier around), then every thread
//     accumulates its outputs over the chunk.
//
//     What is staged are not the raw operands but the values of the body
//     that depend only on the A (or only on the B) inputs and are used by
//     the rest of the body: an MXINT8 weight read by the contraction is
//     restored once per block while staging, and every row of the block
//     reads the restored value from workgroup memory.
//
// Each output keeps its reduction order: the reduction dimensions run
// lexicographically (the innermost fastest), as in the loop nest of
// convert-linalg-to-parallel-loops, and the body is the op's body: the
// results are bit-identical to the one-thread-per-output kernels.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
// Threads of an elementwise block.
constexpr int64_t elementwiseThreads = 256;
// Blocks that fill the T4 (40 SMs, two blocks each).
constexpr int64_t fullGrid = 80;
// Workgroup memory of one block (the static limit of a CUDA block).
constexpr int64_t workgroupBytes = 48 << 10;

Value index(OpBuilder &b, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(b, loc, value);
}

Value product(OpBuilder &b, Location loc, ArrayRef<Value> sizes) {
  Value result = index(b, loc, 1);
  for (Value size : sizes)
    result = arith::MulIOp::create(b, loc, result, size);
  return result;
}

int64_t staticProduct(ArrayRef<int64_t> sizes) {
  int64_t result = 1;
  for (int64_t size : sizes) {
    if (ShapedType::isDynamic(size))
      return ShapedType::kDynamic;
    result *= size;
  }
  return result;
}

// The indices of `linear` in a row-major space of `sizes`.
SmallVector<Value> delinearize(OpBuilder &b, Location loc, Value linear,
                               ArrayRef<Value> sizes) {
  SmallVector<Value> indices(sizes.size());
  for (int64_t d = sizes.size() - 1; d > 0; --d) {
    indices[d] = arith::RemUIOp::create(b, loc, linear, sizes[d]);
    linear = arith::DivUIOp::create(b, loc, linear, sizes[d]);
  }
  if (!sizes.empty())
    indices[0] = linear;
  return indices;
}

// The entries of `all` at the dimensions of `group`.
template <typename T>
SmallVector<T> pick(ArrayRef<unsigned> group, ArrayRef<T> all) {
  SmallVector<T> picked;
  for (unsigned d : group)
    picked.push_back(all[d]);
  return picked;
}

Value less(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, lhs, rhs);
}

// The operand indices of `map` at the iteration point `point`.
SmallVector<Value> apply(OpBuilder &b, Location loc, AffineMap map,
                         ArrayRef<Value> point) {
  SmallVector<Value> indices;
  for (AffineExpr expr : map.getResults()) {
    if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
      indices.push_back(point[dim.getPosition()]);
      continue;
    }
    if (auto constant = dyn_cast<AffineConstantExpr>(expr)) {
      indices.push_back(index(b, loc, constant.getValue()));
      continue;
    }
    indices.push_back(affine::AffineApplyOp::create(
        b, loc, AffineMap::get(map.getNumDims(), 0, expr), point));
  }
  return indices;
}

// The value of input operand `operand` at `point`: a load, or the scalar.
Value read(OpBuilder &b, Location loc, linalg::GenericOp op, OpOperand *operand,
           ArrayRef<Value> point) {
  Value value = operand->get();
  if (!isa<MemRefType>(value.getType()))
    return value;
  return memref::LoadOp::create(
      b, loc, value, apply(b, loc, op.getMatchingIndexingMap(operand), point));
}

// One thread per iteration.
void elementwise(OpBuilder &b, linalg::GenericOp op) {
  Location loc = op.getLoc();
  SmallVector<Value> sizes;
  for (Range range :
       cast<linalg::LinalgOp>(op.getOperation()).createLoopRanges(b, loc))
    sizes.push_back(getValueOrCreateConstantIndexOp(b, loc, range.size));
  Value total = product(b, loc, sizes);
  Value threads = index(b, loc, elementwiseThreads);
  Value one = index(b, loc, 1);
  Value blocks = arith::CeilDivUIOp::create(b, loc, total, threads);
  auto launch =
      gpu::LaunchOp::create(b, loc, blocks, one, one, threads, one, one);
  OpBuilder inner = OpBuilder::atBlockBegin(&launch.getBody().front());
  Value linear = arith::AddIOp::create(
      inner, loc,
      arith::MulIOp::create(inner, loc, launch.getBlockIds().x, threads),
      launch.getThreadIds().x);
  scf::IfOp::create(
      inner, loc, less(inner, loc, linear, total),
      [&](OpBuilder &b, Location loc) {
        SmallVector<Value> point = delinearize(b, loc, linear, sizes);
        Block &body = op.getRegion().front();
        IRMapping mapping;
        for (OpOperand *operand : op.getDpsInputOperands())
          mapping.map(op.getMatchingBlockArgument(operand),
                      read(b, loc, op, operand, point));
        for (OpOperand &operand : op.getDpsInitsMutable())
          if (op.payloadUsesValueFromOperand(&operand))
            mapping.map(op.getMatchingBlockArgument(&operand),
                        read(b, loc, op, &operand, point));
        for (Operation &nested : body.without_terminator()) {
          if (auto position = dyn_cast<linalg::IndexOp>(nested)) {
            mapping.map(position.getResult(), point[position.getDim()]);
            continue;
          }
          b.clone(nested, mapping);
        }
        for (OpOperand &operand : op.getDpsInitsMutable())
          memref::StoreOp::create(
              b, loc,
              mapping.lookupOrDefault(body.getTerminator()->getOperand(
                  operand.getOperandNumber() - op.getNumDpsInputs())),
              operand.get(),
              apply(b, loc, op.getMatchingIndexingMap(&operand), point));
        scf::YieldOp::create(b, loc);
      });
  gpu::TerminatorOp::create(inner, loc);
}

// The side of a body value: the inputs it depends on.
enum Side { Free = 0, SideA = 1, SideB = 2, Core = 3 };

struct Tiles {
  int64_t tx, ty, tm, tn, bk;
};

// The block shape for M x N outputs in `batch` batches.
Tiles choose(int64_t m, int64_t n, int64_t batch) {
  auto blocks = [&](int64_t rows, int64_t columns) -> int64_t {
    if (ShapedType::isDynamic(m) || ShapedType::isDynamic(n) ||
        ShapedType::isDynamic(batch))
      return fullGrid;
    return llvm::divideCeil(m, rows) * llvm::divideCeil(n, columns) * batch;
  };
  if (!ShapedType::isDynamic(m) && m <= 8) {
    int64_t tx = 256;
    while (tx > 32 && blocks(m, tx) < fullGrid)
      tx /= 2;
    return {tx, 1, m, 1, tx > 128 ? 16 : 32};
  }
  if (!ShapedType::isDynamic(n) && n <= 8) {
    int64_t ty = 256;
    while (ty > 32 && blocks(ty, n) < fullGrid)
      ty /= 2;
    return {1, ty, 1, n, ty > 128 ? 16 : 32};
  }
  int64_t t = 4;
  while (t > 1 && blocks(16 * t, 16 * t) < fullGrid)
    t /= 2;
  return {16, 16, t, t, 32};
}

LogicalResult contraction(OpBuilder &b, linalg::GenericOp op) {
  Location loc = op.getLoc();
  MLIRContext *context = op.getContext();
  // Every output (an argmax has two) is written at the same point.
  AffineMap outputMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  for (OpOperand &output : op.getDpsInitsMutable())
    if (op.getMatchingIndexingMap(&output) != outputMap)
      return op->emitOpError("writes its outputs at different points");
  int64_t outputs = op.getNumDpsInits();
  SmallVector<utils::IteratorType> iterators = op.getIteratorTypesArray();
  unsigned dims = iterators.size();
  SmallVector<unsigned> parallel, reduction;
  for (unsigned d = 0; d < dims; ++d)
    (iterators[d] == utils::IteratorType::parallel ? parallel : reduction)
        .push_back(d);
  // The output indexes each parallel loop once, in some order, and
  // constants (a reduced dimension kept as size 1).
  SmallVector<unsigned> outputDims;
  for (AffineExpr expr : outputMap.getResults()) {
    if (isa<AffineConstantExpr>(expr))
      continue;
    auto dim = dyn_cast<AffineDimExpr>(expr);
    if (!dim || iterators[dim.getPosition()] != utils::IteratorType::parallel ||
        llvm::is_contained(outputDims, dim.getPosition()))
      return op->emitOpError("writes no permutation of its parallel loops");
    outputDims.push_back(dim.getPosition());
  }
  if (outputDims.size() != parallel.size())
    return op->emitOpError("writes no permutation of its parallel loops");
  // The output's innermost dimension (none for a reduction to a scalar).
  std::optional<unsigned> innermost;
  if (!outputDims.empty())
    innermost = outputDims.back();

  // The A and B inputs and the dimension groups.
  auto reads = [&](OpOperand *operand, unsigned d) {
    return isa<MemRefType>(operand->get().getType()) &&
           op.getMatchingIndexingMap(operand).isFunctionOfDim(d);
  };
  SmallVector<Side> inputSide;
  for (OpOperand *operand : op.getDpsInputOperands()) {
    if (!isa<MemRefType>(operand->get().getType()) ||
        cast<MemRefType>(operand->get().getType()).getRank() == 0)
      inputSide.push_back(Free);
    else
      inputSide.push_back(innermost && reads(operand, *innermost) ? SideB
                                                                  : SideA);
  }
  auto readBy = [&](Side side, unsigned d) {
    for (OpOperand *operand : op.getDpsInputOperands())
      if (inputSide[operand->getOperandNumber()] == side && reads(operand, d))
        return true;
    return false;
  };
  // Each group in output order (its innermost dimension fastest).
  SmallVector<unsigned> groupM, groupN, groupBatch;
  for (unsigned d : outputDims) {
    bool a = readBy(SideA, d), bb = readBy(SideB, d);
    if (a && !bb)
      groupM.push_back(d);
    else if (bb && !a)
      groupN.push_back(d);
    else
      groupBatch.push_back(d);
  }

  // The sides of the body values.
  Block &body = op.getRegion().front();
  DenseMap<Value, Side> side;
  for (OpOperand *operand : op.getDpsInputOperands())
    side[op.getMatchingBlockArgument(operand)] =
        inputSide[operand->getOperandNumber()];
  for (OpOperand &output : op.getDpsInitsMutable())
    side[op.getMatchingBlockArgument(&output)] = Core;
  auto sideOf = [&](Value value) {
    auto it = side.find(value);
    return it == side.end() ? Free : it->second;
  };
  for (Operation &nested : body.without_terminator()) {
    // A loop index is read where the outputs accumulate.
    int joined = isa<linalg::IndexOp>(nested) ? Core : Free;
    for (Value operand : nested.getOperands())
      joined |= sideOf(operand);
    for (Value result : nested.getResults())
      side[result] = Side(joined);
  }
  ValueRange yielded = body.getTerminator()->getOperands();
  for (Value value : yielded)
    if (sideOf(value) != Core)
      return op->emitOpError("yields no accumulation of its output");
  // The staged values: A or B values that the core of the body uses.
  SetVector<Value> stagedA, stagedB;
  auto stage = [&](Value value) {
    if (sideOf(value) == SideA)
      stagedA.insert(value);
    if (sideOf(value) == SideB)
      stagedB.insert(value);
  };
  for (Operation &nested : body.without_terminator())
    if (sideOf(nested.getResult(0)) == Core)
      for (Value operand : nested.getOperands())
        stage(operand);

  // Extents.
  SmallVector<Value> sizes;
  for (Range range :
       cast<linalg::LinalgOp>(op.getOperation()).createLoopRanges(b, loc))
    sizes.push_back(getValueOrCreateConstantIndexOp(b, loc, range.size));
  SmallVector<int64_t> staticSizes = op.getStaticLoopRanges();
  SmallVector<Value> sizesM = pick(groupM, ArrayRef<Value>(sizes));
  SmallVector<Value> sizesN = pick(groupN, ArrayRef<Value>(sizes));
  SmallVector<Value> sizesBatch = pick(groupBatch, ArrayRef<Value>(sizes));
  SmallVector<Value> sizesR = pick(reduction, ArrayRef<Value>(sizes));
  Value totalM = product(b, loc, sizesM);
  Value totalN = product(b, loc, sizesN);
  Value totalBatch = product(b, loc, sizesBatch);
  Value totalR = product(b, loc, sizesR);
  int64_t staticM = staticProduct(pick(groupM, ArrayRef<int64_t>(staticSizes)));
  int64_t staticN = staticProduct(pick(groupN, ArrayRef<int64_t>(staticSizes)));
  int64_t staticBatch =
      staticProduct(pick(groupBatch, ArrayRef<int64_t>(staticSizes)));
  int64_t staticR =
      staticProduct(pick(reduction, ArrayRef<int64_t>(staticSizes)));
  Tiles tiles = choose(staticM, staticN, staticBatch);
  int64_t bm = tiles.ty * tiles.tm, bn = tiles.tx * tiles.tn, bk = tiles.bk;
  if (!ShapedType::isDynamic(staticBatch) && staticBatch > 65535)
    return op->emitOpError("has more batches than a grid holds");
  if (!ShapedType::isDynamic(staticM) && llvm::divideCeil(staticM, bm) > 65535)
    return op->emitOpError("has more row tiles than a grid holds");

  // Workgroup memory: BK rows of BM (BN) values, padded by one against bank
  // conflicts.
  auto workgroup =
      gpu::AddressSpaceAttr::get(context, gpu::AddressSpace::Workgroup);
  SmallVector<Type> buffers;
  int64_t bytes = 0;
  auto buffer = [&](Value value, int64_t columns) {
    Type type = value.getType();
    buffers.push_back(MemRefType::get({bk, columns}, type,
                                      MemRefLayoutAttrInterface{}, workgroup));
    bytes +=
        bk * columns *
        (type.isIndex() ? 8
                        : llvm::divideCeil(type.getIntOrFloatBitWidth(), 8));
  };
  for (Value value : stagedA)
    buffer(value, bm + 1);
  for (Value value : stagedB)
    buffer(value, bn + 1);
  if (bytes > workgroupBytes)
    return op->emitOpError("stages more than the workgroup memory of a block");

  Value one = index(b, loc, 1);
  auto launch = gpu::LaunchOp::create(
      b, loc, arith::CeilDivUIOp::create(b, loc, totalN, index(b, loc, bn)),
      arith::CeilDivUIOp::create(b, loc, totalM, index(b, loc, bm)), totalBatch,
      index(b, loc, tiles.tx), index(b, loc, tiles.ty), one, nullptr, nullptr,
      {}, nullptr, buffers);
  ArrayRef<BlockArgument> memories = launch.getWorkgroupAttributionBBArgs();
  OpBuilder k = OpBuilder::atBlockBegin(&launch.getBody().front());
  Value tx = launch.getThreadIds().x, ty = launch.getThreadIds().y;
  Value m0 =
      arith::MulIOp::create(k, loc, launch.getBlockIds().y, index(k, loc, bm));
  Value n0 =
      arith::MulIOp::create(k, loc, launch.getBlockIds().x, index(k, loc, bn));
  SmallVector<Value> batchIndices =
      delinearize(k, loc, launch.getBlockIds().z, sizesBatch);
  Value zero = index(k, loc, 0);
  Value thread = arith::AddIOp::create(
      k, loc, arith::MulIOp::create(k, loc, ty, index(k, loc, tiles.tx)), tx);
  Value threads = index(k, loc, tiles.tx * tiles.ty);

  // The iteration point of linear row m, column n and reduction r (each
  // optional: a missing group indexes 0).
  auto point = [&](OpBuilder &b, Value m, Value n, Value r) {
    SmallVector<Value> result(dims, zero);
    for (auto [d, value] : llvm::zip(groupBatch, batchIndices))
      result[d] = value;
    auto place = [&](ArrayRef<unsigned> group, ArrayRef<Value> groupSizes,
                     Value linear) {
      if (!linear)
        return;
      for (auto [d, value] :
           llvm::zip(group, delinearize(b, loc, linear, groupSizes)))
        result[d] = value;
    };
    place(groupM, sizesM, m);
    place(groupN, sizesN, n);
    place(reduction, sizesR, r);
    return result;
  };
  // The free values of the body (scalars, rank-0 inputs).
  IRMapping freeValues;
  for (OpOperand *operand : op.getDpsInputOperands())
    if (inputSide[operand->getOperandNumber()] == Free)
      freeValues.map(
          op.getMatchingBlockArgument(operand),
          isa<MemRefType>(operand->get().getType())
              ? memref::LoadOp::create(k, loc, operand->get(), ValueRange{})
                    .getResult()
              : operand->get());
  // Whether the staged loads of a side are contiguous along the reduction:
  // its first input indexes its innermost operand dimension by the innermost
  // reduction loop.
  auto alongReduction = [&](Side wanted) {
    for (OpOperand *operand : op.getDpsInputOperands())
      if (inputSide[operand->getOperandNumber()] == wanted)
        return !reduction.empty() && op.getMatchingIndexingMap(operand)
                                         .getResults()
                                         .back()
                                         .isFunctionOfDim(reduction.back());
    return false;
  };

  auto rowOf = [&](OpBuilder &b, int64_t i) {
    return arith::AddIOp::create(
        b, loc, m0,
        arith::AddIOp::create(b, loc, ty, index(b, loc, i * tiles.ty)));
  };
  auto columnOf = [&](OpBuilder &b, int64_t j) {
    return arith::AddIOp::create(
        b, loc, n0,
        arith::AddIOp::create(b, loc, tx, index(b, loc, j * tiles.tx)));
  };
  auto inside = [&](OpBuilder &b, Value m, Value n) {
    return arith::AndIOp::create(b, loc, less(b, loc, m, totalM),
                                 less(b, loc, n, totalN));
  };

  // The accumulators (output o of row i, column j at (i * TN + j) * outputs
  // + o) start from the outputs.
  SmallVector<Value> initial;
  for (int64_t i = 0; i < tiles.tm; ++i)
    for (int64_t j = 0; j < tiles.tn; ++j) {
      Value m = rowOf(k, i), n = columnOf(k, j);
      auto start = scf::IfOp::create(
          k, loc, inside(k, m, n),
          [&](OpBuilder &b, Location loc) {
            SmallVector<Value> values;
            for (OpOperand &output : op.getDpsInitsMutable())
              values.push_back(
                  read(b, loc, op, &output, point(b, m, n, Value())));
            scf::YieldOp::create(b, loc, values);
          },
          [&](OpBuilder &b, Location loc) {
            SmallVector<Value> values;
            for (Value output : op.getDpsInits())
              values.push_back(arith::ConstantOp::create(
                  b, loc,
                  cast<TypedAttr>(b.getZeroAttr(
                      cast<MemRefType>(output.getType()).getElementType()))));
            scf::YieldOp::create(b, loc, values);
          });
      llvm::append_range(initial, start.getResults());
    }

  // Stage the chunk at r0 of one side: `count` rows of the side's group.
  auto stageSide = [&](OpBuilder &b, Value r0, Side wanted,
                       const SetVector<Value> &staged,
                       ArrayRef<BlockArgument> targets, int64_t count,
                       Value base, Value total) {
    if (staged.empty())
      return;
    bool kFast = alongReduction(wanted);
    Value countValue = index(b, loc, count), bkValue = index(b, loc, bk);
    scf::ForOp::create(
        b, loc, thread, index(b, loc, count * bk), threads, ValueRange{},
        [&](OpBuilder &b, Location loc, Value e, ValueRange) {
          Value kk = kFast
                         ? Value(arith::RemUIOp::create(b, loc, e, bkValue))
                         : Value(arith::DivUIOp::create(b, loc, e, countValue));
          Value row =
              kFast ? Value(arith::DivUIOp::create(b, loc, e, bkValue))
                    : Value(arith::RemUIOp::create(b, loc, e, countValue));
          Value linear = arith::AddIOp::create(b, loc, base, row);
          Value r = arith::AddIOp::create(b, loc, r0, kk);
          scf::IfOp::create(
              b, loc,
              arith::AndIOp::create(b, loc, less(b, loc, linear, total),
                                    less(b, loc, r, totalR)),
              [&](OpBuilder &b, Location loc) {
                SmallVector<Value> at = wanted == SideA
                                            ? point(b, linear, Value(), r)
                                            : point(b, Value(), linear, r);
                IRMapping mapping = freeValues;
                for (OpOperand *operand : op.getDpsInputOperands())
                  if (inputSide[operand->getOperandNumber()] == wanted)
                    mapping.map(op.getMatchingBlockArgument(operand),
                                read(b, loc, op, operand, at));
                for (Operation &nested : body.without_terminator())
                  if (sideOf(nested.getResult(0)) == wanted ||
                      sideOf(nested.getResult(0)) == Free)
                    b.clone(nested, mapping);
                for (auto [value, target] : llvm::zip(staged, targets))
                  memref::StoreOp::create(b, loc,
                                          mapping.lookupOrDefault(value),
                                          target, ValueRange{kk, row});
                scf::YieldOp::create(b, loc);
              });
          scf::YieldOp::create(b, loc);
        });
  };

  scf::ForOp steps;
  auto chunks = scf::ForOp::create(
      k, loc, zero, totalR, index(k, loc, bk), initial,
      [&](OpBuilder &b, Location loc, Value r0, ValueRange accumulators) {
        gpu::BarrierOp::create(b, loc);
        stageSide(b, r0, SideA, stagedA, memories.take_front(stagedA.size()),
                  bm, m0, totalM);
        stageSide(b, r0, SideB, stagedB, memories.drop_front(stagedA.size()),
                  bn, n0, totalN);
        gpu::BarrierOp::create(b, loc);
        // A whole chunk when BK divides the reduction: a constant trip count
        // that LLVM unrolls.
        Value length = !ShapedType::isDynamic(staticR) && staticR % bk == 0
                           ? index(b, loc, bk)
                           : Value(arith::MinUIOp::create(
                                 b, loc, index(b, loc, bk),
                                 arith::SubIOp::create(b, loc, totalR, r0)));
        steps = scf::ForOp::create(
            b, loc, zero, length, one, accumulators,
            [&](OpBuilder &b, Location loc, Value kk, ValueRange current) {
              // Row i's staged A values and column j's staged B values.
              SmallVector<SmallVector<Value>> rowValues(tiles.tm),
                  columnValues(tiles.tn);
              for (int64_t i = 0; i < tiles.tm; ++i)
                for (auto [n, value] : llvm::enumerate(stagedA))
                  rowValues[i].push_back(memref::LoadOp::create(
                      b, loc, memories[n],
                      ValueRange{
                          kk, arith::AddIOp::create(
                                  b, loc, ty, index(b, loc, i * tiles.ty))}));
              for (int64_t j = 0; j < tiles.tn; ++j)
                for (auto [n, value] : llvm::enumerate(stagedB))
                  columnValues[j].push_back(memref::LoadOp::create(
                      b, loc, memories[stagedA.size() + n],
                      ValueRange{
                          kk, arith::AddIOp::create(
                                  b, loc, tx, index(b, loc, j * tiles.tx))}));
              SmallVector<Value> next;
              for (int64_t i = 0; i < tiles.tm; ++i)
                for (int64_t j = 0; j < tiles.tn; ++j) {
                  IRMapping mapping = freeValues;
                  for (auto [n, value] : llvm::enumerate(stagedA))
                    mapping.map(value, rowValues[i][n]);
                  for (auto [n, value] : llvm::enumerate(stagedB))
                    mapping.map(value, columnValues[j][n]);
                  for (auto [o, output] :
                       llvm::enumerate(op.getDpsInitsMutable()))
                    mapping.map(op.getMatchingBlockArgument(&output),
                                current[(i * tiles.tn + j) * outputs + o]);
                  SmallVector<Value> at;
                  for (Operation &nested : body.without_terminator()) {
                    if (auto position = dyn_cast<linalg::IndexOp>(nested)) {
                      if (at.empty())
                        at = point(b, rowOf(b, i), columnOf(b, j),
                                   arith::AddIOp::create(b, loc, r0, kk));
                      mapping.map(position.getResult(), at[position.getDim()]);
                      continue;
                    }
                    Side s = sideOf(nested.getResult(0));
                    if (s == Core || s == Free)
                      b.clone(nested, mapping);
                  }
                  for (Value value : yielded)
                    next.push_back(mapping.lookupOrDefault(value));
                }
              scf::YieldOp::create(b, loc, next);
            });
        scf::YieldOp::create(b, loc, steps.getResults());
      });

  // Store the outputs.
  for (int64_t i = 0; i < tiles.tm; ++i)
    for (int64_t j = 0; j < tiles.tn; ++j) {
      Value m = rowOf(k, i), n = columnOf(k, j);
      scf::IfOp::create(
          k, loc, inside(k, m, n), [&](OpBuilder &b, Location loc) {
            SmallVector<Value> at =
                apply(b, loc, outputMap, point(b, m, n, Value()));
            for (auto [o, output] : llvm::enumerate(op.getDpsInits()))
              memref::StoreOp::create(
                  b, loc, chunks.getResult((i * tiles.tn + j) * outputs + o),
                  output, at);
            scf::YieldOp::create(b, loc);
          });
    }
  gpu::TerminatorOp::create(k, loc);
  // Unroll whole chunks: the loads of later steps issue before the
  // accumulations of earlier ones complete.
  if (!ShapedType::isDynamic(staticR) && staticR % bk == 0 &&
      failed(loopUnrollByFactor(
          steps,
          std::min<int64_t>(
              bk, std::max<int64_t>(1, 32 / (tiles.tm * tiles.tn * outputs))))))
    return op->emitOpError("cannot unroll its reduction chunk");
  return success();
}

class LinalgToGPUKernelsPass
    : public PassWrapper<LinalgToGPUKernelsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToGPUKernelsPass)

  StringRef getArgument() const final { return "linalg-to-gpu-kernels"; }
  StringRef getDescription() const final {
    return "Map every linalg.generic on buffers to a thread-block kernel";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect, gpu::GPUDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    SmallVector<linalg::GenericOp> ops;
    getOperation().walk([&](linalg::GenericOp op) { ops.push_back(op); });
    for (linalg::GenericOp op : ops) {
      if (!op.hasPureBufferSemantics()) {
        op->emitOpError("is not on buffers");
        return signalPassFailure();
      }
      OpBuilder b(op);
      if (op.getNumReductionLoops() == 0)
        elementwise(b, op);
      else if (failed(contraction(b, op)))
        return signalPassFailure();
      op->erase();
    }
  }
};
} // namespace

namespace mlir::nv {
void registerLinalgToGPUKernelsPass() {
  PassRegistration<LinalgToGPUKernelsPass>();
}
} // namespace mlir::nv

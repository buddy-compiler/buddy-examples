//===- CoreDialect.cpp - Subgraphs on the hardware threads of a core ------===//

#include "Core/CoreDialect.h"
#include "Core/CoreOps.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace ::buddy::core;

#include "CoreDialect.cpp.inc"
#define GET_OP_CLASSES
#include "Core.cpp.inc"

// Bufferized, an operation reads the buffers of its inputs and writes those
// of its destinations.
static void getBufferEffects(
    DestinationStyleOpInterface op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (OpOperand &operand : op->getOpOperands()) {
    if (!isa<MemRefType>(operand.get().getType()))
      continue;
    if (op.isDpsInit(&operand))
      effects.emplace_back(MemoryEffects::Write::get(), &operand, /*stage=*/0,
                           /*effectOnFullRegion=*/true,
                           SideEffects::DefaultResource::get());
    else
      effects.emplace_back(MemoryEffects::Read::get(), &operand, /*stage=*/0,
                           /*effectOnFullRegion=*/true,
                           SideEffects::DefaultResource::get());
  }
}

void LaunchOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getBufferEffects(*this, effects);
}

LogicalResult LaunchOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto callee =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getCalleeAttr());
  if (!callee)
    return emitOpError("launches ") << getCallee() << ", no function";
  // Bufferized, the callee's signature follows bufferization of its module.
  if (getResults().empty())
    return success();
  FunctionType type = callee.getFunctionType();
  if (TypeRange(type.getInputs()) != getOperandTypes() ||
      TypeRange(type.getResults()) != getResults().getTypes())
    return emitOpError("does not match the type of ") << getCallee();
  return success();
}

// A launch reads values of its thread: transferred to it, or results of its
// launches.
LogicalResult LaunchOp::verify() {
  for (Value input : getInputs()) {
    if (!isa<RankedTensorType>(input.getType()))
      continue;
    Operation *def = input.getDefiningOp();
    if (auto transfer = dyn_cast_or_null<TransferOp>(def);
        transfer && transfer.getTo() == getThread())
      continue;
    if (auto launch = dyn_cast_or_null<LaunchOp>(def);
        launch && launch.getThread() == getThread())
      continue;
    return emitOpError("reads a value that is not of thread ") << getThread();
  }
  return success();
}

void TransferOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getBufferEffects(*this, effects);
}

LogicalResult TransferOp::verify() {
  if (!getFrom().has_value() && !getTo().has_value())
    return emitOpError("moves a value from the core CPU to itself");
  if (getFrom().has_value() && getFrom() == getTo())
    return emitOpError("moves a value from a thread to itself");
  return success();
}

namespace {
// A graph with launches and transfers inlines into its caller: a launch
// names its subgraph by symbol.
struct Inliner : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *op, Region *region, bool wouldBeCloned,
                       IRMapping &mapping) const final {
    return true;
  }
};

// Both operations read their inputs and write their destinations, which
// their results are.
template <typename Op>
struct Bufferization
    : public BufferizableOpInterface::ExternalModel<Bufferization<Op>, Op> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &operand,
                              const AnalysisState &state) const {
    return !cast<DestinationStyleOpInterface>(op).isDpsInit(&operand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &operand,
                               const AnalysisState &state) const {
    return cast<DestinationStyleOpInterface>(op).isDpsInit(&operand);
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &operand,
                                      const AnalysisState &state) const {
    auto dps = cast<DestinationStyleOpInterface>(op);
    if (!dps.isDpsInit(&operand))
      return {};
    return {{dps.getTiedOpResult(&operand), BufferRelation::Equivalent}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    SmallVector<Value> buffers;
    for (Value value : op->getOperands()) {
      FailureOr<Value> buffer = getBuffer(rewriter, value, options, state);
      if (failed(buffer))
        return failure();
      buffers.push_back(*buffer);
    }
    rewriter.setInsertionPoint(op);
    rewriter.create(op->getLoc(), op->getName().getIdentifier(), buffers,
                    TypeRange{}, op->getAttrs());
    replaceOpWithBufferizedValues(
        rewriter, op,
        ValueRange(buffers).take_back(
            cast<DestinationStyleOpInterface>(op).getNumDpsInits()));
    return success();
  }
};
} // namespace

void CoreDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Core.cpp.inc"
      >();
  addInterfaces<Inliner>();
  LaunchOp::attachInterface<Bufferization<LaunchOp>>(*getContext());
  TransferOp::attachInterface<Bufferization<TransferOp>>(*getContext());
}

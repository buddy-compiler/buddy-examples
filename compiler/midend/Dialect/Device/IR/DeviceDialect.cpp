//===- DeviceDialect.cpp - Graphs on the cores of a device ----------------===//

#include "Device/DeviceDialect.h"
#include "Device/DeviceOps.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace ::buddy::device;

#include "DeviceDialect.cpp.inc"
#define GET_OP_CLASSES
#include "Device.cpp.inc"

LogicalResult LaunchOp::verify() {
  if (getCoreClass().has_value() == getCore().has_value())
    return emitOpError("names either the class of its core or its core");
  return success();
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
  if (TypeRange(type.getInputs()) != getInputs().getTypes() ||
      TypeRange(type.getResults()) != getResults().getTypes())
    return emitOpError("does not match the type of ") << getCallee();
  return success();
}

namespace {
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

void DeviceDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Device.cpp.inc"
      >();
  LaunchOp::attachInterface<Bufferization<LaunchOp>>(*getContext());
  TransferOp::attachInterface<Bufferization<TransferOp>>(*getContext());
}

//===- BuckyballDialect.cpp - Buckyball dialect ---------------------------===//

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace ::buddy::buckyball;

#include "BuckyballDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Buckyball.cpp.inc"

void BuckyballDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Buckyball.cpp.inc"
      >();
}

// A Ball reads at most two virtual banks.
LogicalResult BankCustomOp::verify() {
  if (getSources().size() > 2)
    return emitOpError("reads at most two source banks");
  return success();
}

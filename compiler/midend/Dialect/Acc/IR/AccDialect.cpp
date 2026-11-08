//===- AccDialect.cpp - Acc dialect ---------------------------------------===//

#include "Acc/AccDialect.h"
#include "Acc/AccOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace ::buddy::acc;

#include "AccDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Acc.cpp.inc"

void AccDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Acc.cpp.inc"
      >();
}

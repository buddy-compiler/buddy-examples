#include "Custom/CustomDialect.h"
#include "Custom/CustomOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace ::buddy::custom;

#include "CustomDialect.cpp.inc"
#define GET_OP_CLASSES
#include "Custom.cpp.inc"

void CustomDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Custom.cpp.inc"
      >();
}

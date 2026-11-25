//===- GvnDialect.cpp - GVN dialect ---------------------------------------===//

#include "GvnOps.h"

using namespace mlir;
using namespace ::gvn;

#include "GvnDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Gvn.cpp.inc"

void GvnDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Gvn.cpp.inc"
      >();
}

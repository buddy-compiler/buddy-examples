//===- FrontendDialect.cpp - Frontend dialect -----------------------------===//

#include "Frontend/FrontendDialect.h"
#include "Frontend/FrontendOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace ::buddy::frontend;

#include "FrontendDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Frontend.cpp.inc"

void FrontendDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Frontend.cpp.inc"
      >();
}

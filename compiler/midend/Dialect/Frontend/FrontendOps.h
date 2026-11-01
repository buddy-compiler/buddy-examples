//===- FrontendOps.h - Frontend operations --------------------------------===//

#ifndef FRONTEND_FRONTENDOPS_H
#define FRONTEND_FRONTENDOPS_H

#include "Frontend/FrontendDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Frontend.h.inc"

#endif // FRONTEND_FRONTENDOPS_H

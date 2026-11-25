//===- GvnOps.h - GVN dialect ---------------------------------------------===//

#ifndef GVN_GVNOPS_H
#define GVN_GVNOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "GvnDialect.h.inc"

#define GET_OP_CLASSES
#include "Gvn.h.inc"

#endif // GVN_GVNOPS_H

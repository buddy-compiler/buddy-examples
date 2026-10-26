#ifndef CUSTOM_CUSTOMOPS_H
#define CUSTOM_CUSTOMOPS_H

#include "Custom/CustomDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Custom.h.inc"

#endif // CUSTOM_CUSTOMOPS_H

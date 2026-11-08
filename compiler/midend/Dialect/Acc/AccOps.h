//===- AccOps.h - Acc operations ------------------------------------------===//

#ifndef ACC_ACCOPS_H
#define ACC_ACCOPS_H

#include "Acc/AccDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Acc.h.inc"

#endif // ACC_ACCOPS_H

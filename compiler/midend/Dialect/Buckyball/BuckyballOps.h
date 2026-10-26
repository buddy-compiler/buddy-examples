//===- BuckyballOps.h - Buckyball operations ------------------------------===//

#ifndef BUCKYBALL_BUCKYBALLOPS_H
#define BUCKYBALL_BUCKYBALLOPS_H

#include "Buckyball/BuckyballDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Buckyball.h.inc"

#endif // BUCKYBALL_BUCKYBALLOPS_H

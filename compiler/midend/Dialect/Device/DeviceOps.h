#ifndef DEVICE_DEVICEOPS_H
#define DEVICE_DEVICEOPS_H

#include "Device/DeviceDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#define GET_OP_CLASSES
#include "Device.h.inc"

#endif // DEVICE_DEVICEOPS_H

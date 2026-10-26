//===- Buckyball.h - Buckyball framework passes ---------------------------===//
//
// assign-physical-banks maps the virtual banks to the banks of the target;
// lower-buckyball lowers the physical operations to Buckyball instructions,
// the custom operations through the Ball of their kind.
//
//===----------------------------------------------------------------------===//

#ifndef BUCKYBALL_CONVERSION_BUCKYBALL_H
#define BUCKYBALL_CONVERSION_BUCKYBALL_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::buddy {

void registerBuckyballPasses();

// A Ball adds the patterns lowering the custom operations of its kinds; it
// registers them when the compiler plugin is loaded.
void addBallLowering(void (*populate)(RewritePatternSet &patterns));

// Emits one Buckyball instruction, rs1 and rs2 with the immediate funct7.
void emitInstruction(OpBuilder &builder, Location loc, int32_t funct7,
                     Value rs1, Value rs2);

Value constantI64(OpBuilder &builder, Location loc, uint64_t value);

// Bits [start, end] of an instruction operand holding `value`.
uint64_t field(uint64_t value, unsigned start, unsigned end);

// The rs1 of a Ball instruction: its source banks, its destination bank,
// whether it reads the old destination, and its iteration count.
uint64_t bankFields(llvm::ArrayRef<int64_t> sources, int64_t destination,
                    bool readDestination, uint64_t iterations);

} // namespace mlir::buddy

#endif // BUCKYBALL_CONVERSION_BUCKYBALL_H

#pragma once
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
namespace mlir {
class RewritePatternSet;
}
namespace mlir::buddy::rvv {
Value source(Value value);
Value expression(Value value);
Operation *operation(Value value, llvm::StringRef name);
bool constant(Value value, double expected);
void populateTensorPatterns(RewritePatternSet &patterns);
} // namespace mlir::buddy::rvv

#pragma once

namespace mlir {
class RewritePatternSet;
}

namespace mlir::buddy {
void registerFusePointwisePass();
void registerFoldUnitTransposePass();
void registerGatherRowsPass();
void populateGatherPatterns(RewritePatternSet &patterns);
void registerRVVKernelsPasses();
void registerRVVMatmulPass();
} // namespace mlir::buddy

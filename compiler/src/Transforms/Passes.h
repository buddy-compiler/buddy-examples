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
void registerRVVPointwisePass();
} // namespace mlir::buddy

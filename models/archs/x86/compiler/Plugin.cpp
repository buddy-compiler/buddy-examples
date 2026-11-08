//===- Plugin.cpp - The x86 backend compiler as a buddy-opt plugin ------===//
//
// buddy-opt --load-dialect-plugin=<so> --load-pass-plugin=<so> loads the
// framework's frontend dialect and the passes the x86 backend owns
// (fuse-mxint8-weights and fuse-elementwise).
//
//===----------------------------------------------------------------------===//

#include "Frontend/FrontendDialect.h"

#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"

namespace mlir::x86 {
void registerFuseMXINT8WeightsPass();
void registerFuseElementwisePass();
} // namespace mlir::x86

extern "C" LLVM_ATTRIBUTE_WEAK mlir::DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "x86", LLVM_VERSION_STRING,
          [](mlir::DialectRegistry *registry) {
            registry->insert<::buddy::frontend::FrontendDialect>();
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "x86", LLVM_VERSION_STRING, [] {
            mlir::x86::registerFuseMXINT8WeightsPass();
            mlir::x86::registerFuseElementwisePass();
          }};
}

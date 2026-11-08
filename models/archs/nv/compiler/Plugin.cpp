//===- Plugin.cpp - The NVIDIA backend compiler as a buddy-opt plugin
//------===//
//
// buddy-opt --load-dialect-plugin=<so> --load-pass-plugin=<so> loads the
// framework's frontend dialect and the passes the NVIDIA backend owns
// (fuse-mxint8-weights, linalg-to-gpu-kernels).
//
//===----------------------------------------------------------------------===//

#include "Frontend/FrontendDialect.h"

#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"

namespace mlir::nv {
void registerFuseMXINT8WeightsPass();
void registerLinalgToGPUKernelsPass();
} // namespace mlir::nv

extern "C" LLVM_ATTRIBUTE_WEAK mlir::DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "NVIDIA", LLVM_VERSION_STRING,
          [](mlir::DialectRegistry *registry) {
            registry->insert<::buddy::frontend::FrontendDialect>();
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "NVIDIA", LLVM_VERSION_STRING, [] {
            mlir::nv::registerFuseMXINT8WeightsPass();
            mlir::nv::registerLinalgToGPUKernelsPass();
          }};
}

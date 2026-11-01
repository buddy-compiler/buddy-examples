//===- Plugin.cpp - The generic midend as a buddy-opt plugin --------------===//
//
// buddy-opt --load-dialect-plugin=<so> --load-pass-plugin=<so> loads the
// frontend dialect and lower-frontend-fake-op for the targets without Balls,
// which compute every frontend operation in software.
//
//===----------------------------------------------------------------------===//

#include "Frontend/FrontendDialect.h"

#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"

namespace mlir::buddy {
void registerLowerFrontendFakeOpPass();
} // namespace mlir::buddy

extern "C" LLVM_ATTRIBUTE_WEAK mlir::DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Generic", LLVM_VERSION_STRING,
          [](mlir::DialectRegistry *registry) {
            registry->insert<::buddy::frontend::FrontendDialect>();
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Generic", LLVM_VERSION_STRING,
          [] { mlir::buddy::registerLowerFrontendFakeOpPass(); }};
}

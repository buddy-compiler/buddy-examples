//===- Plugin.cpp - The GVN stimulus compiler as a buddy-opt plugin -------===//
//
// buddy-opt --load-dialect-plugin=<so> --load-pass-plugin=<so>, next to the
// plugin of the hart compiler (NPU side) or the generic one (GPU side).
//
//===----------------------------------------------------------------------===//

#include "GvnOps.h"

#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"

namespace gvn {
void registerInsertChecksPass();
void registerSlicePass();
void registerSliceOpPass();
void registerExposePass();
void registerForwardResultsPass();
void registerImportPass();
void registerGpuProgramPass();
void registerHostShapesPass();
void registerManagedMemoryPass();
void registerPackedEntryPass();
} // namespace gvn

extern "C" LLVM_ATTRIBUTE_WEAK mlir::DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "GVN", LLVM_VERSION_STRING,
          [](mlir::DialectRegistry *registry) {
            registry->insert<::gvn::GvnDialect>();
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "GVN", LLVM_VERSION_STRING, [] {
            gvn::registerInsertChecksPass();
            gvn::registerSlicePass();
            gvn::registerSliceOpPass();
            gvn::registerExposePass();
            gvn::registerForwardResultsPass();
            gvn::registerImportPass();
            gvn::registerGpuProgramPass();
            gvn::registerHostShapesPass();
            gvn::registerManagedMemoryPass();
            gvn::registerPackedEntryPass();
          }};
}

//===- Plugin.cpp - The Buckyball compiler as a buddy-opt plugin ----------===//
//
// buddy-opt --load-dialect-plugin=<so> --load-pass-plugin=<so> loads the
// Buckyball dialect and the framework passes. The lowerings of the Balls and
// the passes of the core register themselves when the library is loaded.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballDialect.h"
#include "Conversion/Buckyball.h"
#include "Custom/CustomDialect.h"

#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"

using namespace mlir;

extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Buckyball", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) {
            registry->insert<::buddy::buckyball::BuckyballDialect,
                             ::buddy::custom::CustomDialect>();
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Buckyball", LLVM_VERSION_STRING,
          [] { mlir::buddy::registerBuckyballPasses(); }};
}

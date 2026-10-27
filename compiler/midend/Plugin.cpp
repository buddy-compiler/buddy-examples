//===- Plugin.cpp - The Buckyball compiler as a buddy-opt plugin ----------===//
//
// buddy-opt --load-dialect-plugin=<so> --load-pass-plugin=<so> loads the
// Buckyball dialect, the dialects of the Balls and the framework passes. The
// Balls and the passes of the core register themselves when the library is
// loaded.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballDialect.h"
#include "Conversion/Buckyball.h"
#include "Custom/CustomDialect.h"

#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"

using namespace mlir;

static SmallVector<void (*)(DialectRegistry &)> &ballDialects() {
  static SmallVector<void (*)(DialectRegistry &)> dialects;
  return dialects;
}

void mlir::buddy::addBallDialect(void (*insert)(DialectRegistry &registry)) {
  ballDialects().push_back(insert);
}

extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Buckyball", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) {
            registry->insert<::buddy::buckyball::BuckyballDialect,
                             ::buddy::custom::CustomDialect>();
            for (auto insert : ballDialects())
              insert(*registry);
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Buckyball", LLVM_VERSION_STRING,
          [] { mlir::buddy::registerBuckyballPasses(); }};
}

//===- Plugin.cpp - The Buckyball compiler as a buddy-opt plugin ----------===//
//
// buddy-opt --load-dialect-plugin=<so> --load-pass-plugin=<so> of any plugin
// of a level loads this library: the acc, device, hart and frontend dialects,
// the dialects of the Balls and the framework passes. The library is loaded
// once however many plugins link it, so the framework passes register once,
// when it loads; the Balls and the passes of the designs register themselves
// when their libraries load.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccDialect.h"
#include "Conversion/Buckyball.h"
#include "Device/DeviceDialect.h"
#include "Frontend/FrontendDialect.h"
#include "Hart/HartDialect.h"

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

void mlir::buddy::registerBallDialects(DialectRegistry &registry) {
  for (auto insert : ballDialects())
    insert(registry);
}

extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {
      MLIR_PLUGIN_API_VERSION, "Buckyball", LLVM_VERSION_STRING,
      [](DialectRegistry *registry) {
        registry->insert<
            ::buddy::acc::AccDialect, ::buddy::device::DeviceDialect,
            ::buddy::hart::HartDialect, ::buddy::frontend::FrontendDialect>();
        mlir::buddy::registerBallDialects(*registry);
      }};
}

static const bool registered = (mlir::buddy::registerBuckyballPasses(), true);

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Buckyball", LLVM_VERSION_STRING, [] {}};
}

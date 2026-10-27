//===- BuckyballTargetRegistry.h - The compiled Buckyball target ----------===//
//
// The banks and ISA of the compiler target the plugin compiles for. The plugin
// reads them from design.pb, named on the buddy-opt command line:
//   --buckyball-design-pb=<design.pb> --buckyball-target=<target>
//
//===----------------------------------------------------------------------===//

#ifndef BUCKYBALL_TARGET_REGISTRY_H
#define BUCKYBALL_TARGET_REGISTRY_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace buckyball_target {

struct BuckyballTargetConfig {
  std::string core;
  int64_t bankNum;
  int64_t bankWidthBits;
  int64_t bankDepth;
  int64_t privateBankMax;
  bool rvvEnabled;
  // Whether a CPU core of the device runs the target's programs, under the OS
  // whose DMA service tracks the banks; otherwise only hardware threads do.
  bool cpuCore;
};

const BuckyballTargetConfig &getBuckyballTarget();
// The funct7 of the one instruction of the target named `mnemonic`.
int32_t getBuckyballFunct7(llvm::StringRef mnemonic);
// A parameter of the Ball implementing the instruction `mnemonic`.
int64_t getBuckyballBallParam(llvm::StringRef mnemonic, llvm::StringRef param);

} // namespace buckyball_target

#endif // BUCKYBALL_TARGET_REGISTRY_H

//===- TargetRegistry.h - The compiled target -----------------------------===//
//
// The banks and ISA of the compiler target the plugin compiles for. The plugin
// reads them from design.pb, named on the buddy-opt command line:
//   --design-pb=<design.pb> --design-target=<target>
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_REGISTRY_H
#define TARGET_REGISTRY_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace target {

struct TargetConfig {
  std::string hart;
  int64_t bankNum;
  int64_t bankWidthBits;
  int64_t bankDepth;
  int64_t privateBankMax;
  bool rvvEnabled;
  bool fp64Enabled;
  int64_t dmaMaxRanges;
};

const TargetConfig &getTarget();
// The funct7 of the one instruction of the target named `mnemonic`.
int32_t getFunct7(llvm::StringRef mnemonic);
// A parameter of the Ball implementing the instruction `mnemonic`.
int64_t getBallParam(llvm::StringRef mnemonic, llvm::StringRef param);

} // namespace target

#endif // TARGET_REGISTRY_H

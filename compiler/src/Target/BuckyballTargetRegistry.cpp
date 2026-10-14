#include "Target/BuckyballTargetRegistry.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#include "BuckyballTargetRegistry.inc"

const buckyball_target::BuckyballTargetConfig &
buckyball_target::getBuckyballTarget() {
  return kBuckyballTarget;
}

int32_t buckyball_target::getBuckyballFunct7(llvm::StringRef mnemonic) {
  for (const BuckyballIsaEntry &entry : kBuckyballTarget.isa) {
    if (entry.mnemonic == mnemonic)
      return entry.funct7;
  }
  report_fatal_error(Twine("Buckyball core ") + kBuckyballTarget.core +
                     " does not define ISA mnemonic " + mnemonic);
}

void buckyball_target::requireBuckyballBall(llvm::StringRef ballName) {
  for (llvm::StringRef enabled : kBuckyballTarget.balls) {
    if (enabled == ballName)
      return;
  }
  report_fatal_error(Twine("Buckyball core ") + kBuckyballTarget.core +
                     " does not enable Ball " + ballName);
}

const buckyball_target::BuckyballBallMapping &
buckyball_target::getBuckyballBallMapping(llvm::StringRef ballName) {
  for (const BuckyballBallMapping &mapping : kBuckyballTarget.ballMappings) {
    if (mapping.name == ballName)
      return mapping;
  }
  report_fatal_error(Twine("Buckyball core ") + kBuckyballTarget.core +
                     " does not map Ball " + ballName);
}

int64_t buckyball_target::getBuckyballBallParam(llvm::StringRef ballName,
                                                llvm::StringRef param) {
  for (const BuckyballBallParam &entry :
       getBuckyballBallMapping(ballName).params) {
    if (entry.name == param)
      return entry.value;
  }
  report_fatal_error(Twine("Buckyball Ball ") + ballName +
                     " has no parameter " + param);
}

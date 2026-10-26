//===- BuckyballTargetRegistry.cpp - The compiled Buckyball target --------===//

#include "Target/BuckyballTargetRegistry.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

#include "BuckyballTargetRegistry.inc"

using namespace buckyball_target;

const BuckyballTargetConfig &buckyball_target::getBuckyballTarget() {
  return kBuckyballTarget;
}

static const BuckyballIsaEntry &instruction(llvm::StringRef mnemonic) {
  const BuckyballIsaEntry *found = nullptr;
  for (const BuckyballIsaEntry &entry : kBuckyballTarget.isa) {
    if (entry.mnemonic != mnemonic)
      continue;
    if (found)
      llvm::report_fatal_error(llvm::Twine("Several Balls of core ") +
                               kBuckyballTarget.core + " implement " +
                               mnemonic);
    found = &entry;
  }
  if (!found)
    llvm::report_fatal_error(llvm::Twine("Core ") + kBuckyballTarget.core +
                             " has no instruction " + mnemonic);
  return *found;
}

int32_t buckyball_target::getBuckyballFunct7(llvm::StringRef mnemonic) {
  return instruction(mnemonic).funct7;
}

int64_t buckyball_target::getBuckyballBallParam(llvm::StringRef mnemonic,
                                                llvm::StringRef param) {
  const BuckyballBallMapping &ball =
      kBuckyballTarget.ballMappings[instruction(mnemonic).ballIndex];
  for (const BuckyballBallParam &entry : ball.params)
    if (entry.name == param)
      return entry.value;
  llvm::report_fatal_error(llvm::Twine("Ball ") + ball.name +
                           " has no parameter " + param);
}

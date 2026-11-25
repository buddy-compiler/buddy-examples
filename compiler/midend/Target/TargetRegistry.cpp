//===- TargetRegistry.cpp - The compiled target ---------------------------===//

#include "Target/TargetRegistry.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"

#include "design.pb.h"

#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>

#include <memory>

using namespace target;

static llvm::cl::opt<std::string>
    designPb("design-pb", llvm::cl::desc("design.pb the target is read from"));
static llvm::cl::opt<std::string>
    targetName("design-target",
               llvm::cl::desc("Compiler target (profile) of design.pb"));

namespace {
struct Target {
  TargetConfig config;
  buckyball::config::ThreadInstance thread;
};
} // namespace

static const Target &loaded() {
  static const Target loaded = [] {
    if (designPb.empty() || targetName.empty())
      llvm::report_fatal_error("the Buckyball compiler needs "
                               "--design-pb and --design-target");
    auto buffer = llvm::MemoryBuffer::getFile(designPb);
    if (!buffer)
      llvm::report_fatal_error(llvm::Twine("cannot read ") + designPb);
    buckyball::config::Design design;
    if (!design.ParseFromArray((*buffer)->getBufferStart(),
                               (*buffer)->getBufferSize()))
      llvm::report_fatal_error(llvm::Twine("cannot parse ") + designPb);
    for (const auto &profile : design.profiles()) {
      if (profile.name() != targetName)
        continue;
      const auto &thread = profile.thread();
      const auto &bank = thread.mem().bank();
      if (thread.gpdomain().enable() && !thread.gpdomain().has_fp64())
        llvm::report_fatal_error("GP target is missing fp64");
      return Target{{thread.pkg(), bank.num(), bank.width(), bank.entries(),
                     thread.frontend().max_private_vbank(),
                     thread.gpdomain().enable(), thread.gpdomain().fp64(),
                     thread.transport().prepared().max_ranges()},
                    thread};
    }
    llvm::report_fatal_error(llvm::Twine(designPb) + " has no target " +
                             targetName);
  }();
  return loaded;
}

const TargetConfig &target::getTarget() { return loaded().config; }

// The Ball implementing the instruction `mnemonic`, and the instruction.
static std::pair<const buckyball::config::BallInstance *,
                 const buckyball::config::BallInstruction *>
instruction(llvm::StringRef mnemonic) {
  const Target &t = loaded();
  std::pair<const buckyball::config::BallInstance *,
            const buckyball::config::BallInstruction *>
      found{nullptr, nullptr};
  for (const auto &ball : t.thread.balldomain().balls())
    for (const auto &entry : ball.instructions()) {
      if (entry.mnemonic() != mnemonic)
        continue;
      if (found.first)
        llvm::report_fatal_error(llvm::Twine("Several Balls of hart ") +
                                 t.config.hart + " implement " + mnemonic);
      found = {&ball, &entry};
    }
  if (!found.first)
    llvm::report_fatal_error(llvm::Twine("Hart ") + t.config.hart +
                             " has no instruction " + mnemonic);
  return found;
}

int32_t target::getFunct7(llvm::StringRef mnemonic) {
  return instruction(mnemonic).second->funct7();
}

int64_t target::getBallParam(llvm::StringRef mnemonic, llvm::StringRef param) {
  namespace pb = google::protobuf;
  const buckyball::config::BallInstance &ball = *instruction(mnemonic).first;
  // The parameters of a Ball are its own message, packed in an Any; the
  // messages of the target's Balls are compiled into this plugin.
  llvm::StringRef type = ball.parameters().type_url();
  type = type.substr(type.rfind('/') + 1);
  const pb::Descriptor *descriptor =
      pb::DescriptorPool::generated_pool()->FindMessageTypeByName(type.str());
  if (!descriptor)
    llvm::report_fatal_error(llvm::Twine("Ball ") + ball.name() +
                             " has parameters of unknown type " + type);
  std::unique_ptr<pb::Message> params(
      pb::MessageFactory::generated_factory()->GetPrototype(descriptor)->New());
  if (!params->ParseFromString(ball.parameters().value()))
    llvm::report_fatal_error(llvm::Twine("cannot parse the parameters of ") +
                             ball.name());
  const pb::FieldDescriptor *field = descriptor->FindFieldByName(param.str());
  if (!field || !params->GetReflection()->HasField(*params, field))
    llvm::report_fatal_error(llvm::Twine("Ball ") + ball.name() +
                             " has no parameter " + param);
  const pb::Reflection *reflection = params->GetReflection();
  switch (field->cpp_type()) {
  case pb::FieldDescriptor::CPPTYPE_INT32:
    return reflection->GetInt32(*params, field);
  case pb::FieldDescriptor::CPPTYPE_INT64:
    return reflection->GetInt64(*params, field);
  case pb::FieldDescriptor::CPPTYPE_UINT32:
    return reflection->GetUInt32(*params, field);
  default:
    llvm::report_fatal_error(llvm::Twine("Ball ") + ball.name() +
                             " parameter " + param + " is not an integer");
  }
}

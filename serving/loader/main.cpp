#include "buddy/runtime/core/ModelManifest.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: load-model MODEL.rax\n";
    return 1;
  }
  try {
    const auto manifest = buddy::runtime::ModelManifest::loadFromRax(argv[1]);
    llvm::json::Array programs;
    for (const auto &code : manifest.codeObjects)
      programs.push_back(llvm::json::Object{
          {"name", code.name}, {"path", code.path}, {"backend", code.backend},
          {"kind", rhal::rax::EnumNameCodeObjectKind(code.kind)}});
    llvm::json::Object resources;
    for (const auto &constant : manifest.constants) {
      if (!resources.try_emplace(constant.name, constant.path).second)
        throw std::runtime_error("duplicate resource: " + constant.name);
    }
    llvm::outs() << llvm::json::Value(llvm::json::Object{
        {"programs", std::move(programs)}, {"resources", std::move(resources)}})
                 << '\n';
  } catch (const std::exception &error) {
    llvm::errs() << error.what() << '\n';
    return 1;
  }
}

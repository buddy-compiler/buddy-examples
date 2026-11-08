#include "SharedLibrary.h"
#include <buddy/Core/Container.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

// SmolVLA as one graph: the VLM runs once over the images, the prompt and the
// robot state; the action expert then integrates the flow from the noise to an
// action chunk. `--dataset` names the test case the importer wrote (inputs and
// the actions PyTorch samples from them); the runner checks its actions
// against them.

namespace {
template <typename T, size_t N>
void load(const std::string &path, MemRef<T, N> &memref) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Cannot open " + path);
  file.read(reinterpret_cast<char *>(memref.getData()),
            memref.getSize() * sizeof(T));
  if (!file || file.peek() != EOF)
    throw std::runtime_error("Size mismatch: " + path);
}

// The argument order of forward.mlir: Dynamo orders the inputs by first use.
using Forward = void (*)(MemRef<float, 3> *actions, MemRef<float, 1> *params,
                         MemRef<float, 5> *images,
                         MemRef<long long, 2> *langMasks,
                         MemRef<float, 2> *state, MemRef<float, 3> *noise,
                         MemRef<long long, 2> *langTokens);

class SmolVLARunner final : public buddy::runtime::InferenceRunner {
public:
  void run(const buddy::runtime::RunConfig &runConfig) override {
    const auto manifest =
        buddy::runtime::ModelManifest::loadFromRax(runConfig.raxPath);
    std::vector<model::SharedLibrary> runtimes;
    for (const auto &path : manifest.dependentSoPaths)
      runtimes.push_back(model::openLibrary(path));
    auto library = model::openLibrary(manifest.soPath);
    auto forward =
        reinterpret_cast<Forward>(dlsym(library.get(), "_mlir_ciface_forward"));
    if (!forward)
      throw std::runtime_error("Missing _mlir_ciface_forward");
    if (runConfig.datasetPath.empty())
      throw std::runtime_error("--dataset is required");
    const std::filesystem::path testcase = runConfig.datasetPath;

    MemRef<float, 1> params({402737376});
    load(manifest.weightPaths.at(0), params);

    MemRef<float, 5> images({3, 1, 3, 512, 512});
    MemRef<long long, 2> langTokens({1, 48});
    MemRef<long long, 2> langMasks({1, 48});
    MemRef<float, 2> state({1, 32});
    MemRef<float, 3> noise({1, 50, 32});
    MemRef<float, 3> reference({1, 50, 32});
    load(testcase / "images.data", images);
    load(testcase / "lang_tokens.data", langTokens);
    load(testcase / "lang_masks.data", langMasks);
    load(testcase / "state.data", state);
    load(testcase / "noise.data", noise);
    load(testcase / "actions.data", reference);

    MemRef<float, 3> actions({1, 50, 32});
    forward(&actions, &params, &images, &langMasks, &state, &noise,
            &langTokens);

    float maxDiff = 0, maxRef = 0;
    for (size_t i = 0; i < actions.getSize(); ++i) {
      maxDiff = std::max(
          maxDiff, std::fabs(actions.getData()[i] - reference.getData()[i]));
      maxRef = std::max(maxRef, std::fabs(reference.getData()[i]));
    }
    std::cout << "actions max |diff| " << maxDiff << " (max |reference| "
              << maxRef << ")" << std::endl;
    if (!runConfig.outputPath.empty()) {
      std::ofstream out(runConfig.outputPath, std::ios::binary);
      out.write(reinterpret_cast<const char *>(actions.getData()),
                actions.getSize() * sizeof(float));
    }
  }
};
} // namespace

extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new SmolVLARunner();
}
extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

#include "SharedLibrary.h"
#include <algorithm>
#include <buddy/Core/Container.h>
#include <buddy/DAP/AudioContainer.h>
#include <buddy/DAP/DAP.h>
#include <buddy/LLM/TextContainer.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
namespace {
using Forward = void (*)(MemRef<float, 3> *, MemRef<float, 1> *,
                         MemRef<float, 3> *, MemRef<size_t, 2> *);
class WhisperRunner final : public buddy::runtime::InferenceRunner {
public:
  void run(const buddy::runtime::RunConfig &config) override {
    const auto manifest =
        buddy::runtime::ModelManifest::loadFromRax(config.raxPath);
    std::vector<model::SharedLibrary> runtimes;
    for (const auto &path : manifest.dependentSoPaths)
      runtimes.push_back(model::openLibrary(path));
    auto library = model::openLibrary(manifest.soPath);
    auto forward =
        reinterpret_cast<Forward>(dlsym(library.get(), "_mlir_ciface_forward"));
    if (!forward)
      throw std::runtime_error("Missing _mlir_ciface_forward: " +
                               std::string(dlerror()));
    std::ifstream weights;
    weights.exceptions(std::ios::failbit | std::ios::badbit);
    weights.open(manifest.weightPaths.at(0), std::ios::binary | std::ios::ate);
    const size_t bytes = weights.tellg();
    if (bytes % sizeof(float))
      throw std::runtime_error("Invalid FP32 parameter size");
    MemRef<float, 1> parameters(std::vector<size_t>{bytes / sizeof(float)});
    weights.seekg(0);
    weights.read(reinterpret_cast<char *>(parameters.getData()), bytes);
    dap::Audio<double, 1> audio(config.audioPath);
    MemRef<float, 3> features({1, 80, 3000});
    dap::whisperPreprocess(&audio, &features);
    MemRef<size_t, 2> tokens({1, 448}, 50258);
    buddy::Text<size_t, 2> generated;
    generated.loadVocab(manifest.vocabPath);
    const auto start = std::chrono::steady_clock::now();
    size_t count = 0;
    while (count < 447 && (config.maxNewTokens == 0 ||
                           count < static_cast<size_t>(config.maxNewTokens))) {
      MemRef<float, 3> logits(std::vector<size_t>{1, 448, 51865}, false, 0);
      forward(&logits, &parameters, &features, &tokens);
      const float *row = logits.getData() + count * 51865;
      const size_t token = std::max_element(row, row + 51865) - row;
      if (token == 50257)
        break;
      tokens.getData()[++count] = token;
      generated.appendTokenIdx(token);
    }
    std::cout << generated.revertWhisper() << '\n';
    if (!config.suppressStats) {
      const std::chrono::duration<double> elapsed =
          std::chrono::steady_clock::now() - start;
      std::cout << "Generated tokens: " << count
                << ", inference time: " << elapsed.count() << "s\n";
    }
  }
};
} // namespace
extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new WhisperRunner();
}
extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

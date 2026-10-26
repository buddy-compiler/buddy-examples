#include "SharedLibrary.h"
#include <algorithm>
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>
#include <buddy/runtime/llm/Sampler.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
namespace {
using Forward = void (*)(MemRef<float, 3> *, MemRef<float, 1> *,
                         MemRef<size_t, 2> *, MemRef<long long, 2> *);
class GemmaRunner final : public buddy::runtime::InferenceRunner {
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
    constexpr size_t length = 32;
    buddy::Text<size_t, 2> input(config.prompt), generated;
    input.tokenizeGemma4(manifest.vocabPath, length);
    generated.loadVocab(manifest.vocabPath);
    size_t used = input.getTokenCnt();
    if (used == 0 || used >= length)
      throw std::runtime_error("Prompt leaves no room in the compiled context");
    MemRef<long long, 2> mask({1, length}, 0);
    for (size_t i = 0; i < used; ++i)
      mask.getData()[i] = 1;
    buddy::Sampler sampler(config.samplerConfig);
    std::vector<int> tokens;
    const auto start = std::chrono::steady_clock::now();
    while (used < length &&
           (config.maxNewTokens == 0 ||
            tokens.size() < static_cast<size_t>(config.maxNewTokens))) {
      MemRef<float, 3> logits(std::vector<size_t>{1, length, 1}, false, 0);
      forward(&logits, &parameters, &input, &mask);
      const size_t vocab = logits.getSizes()[2];
      const int token =
          sampler.sample(logits.getData() + (used - 1) * vocab, vocab, tokens);
      if (token == 1 || token == 106)
        break;
      tokens.push_back(token);
      generated.appendTokenIdx(token);
      input.getData()[used] = token;
      mask.getData()[used++] = 1;
    }
    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start;
    std::cout << generated.revertGemma4() << '\n';
    if (!config.suppressStats)
      std::cout << "Generated tokens: " << tokens.size()
                << ", inference time: " << elapsed.count() << "s\n";
  }
};

} // namespace
extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new GemmaRunner();
}
extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

#include "SharedLibrary.h"
#include "cache.h"
#include <algorithm>
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>
#include <buddy/runtime/llm/Sampler.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

namespace {
struct Result {
  MemRef<float, 3> logits{std::vector<size_t>{1, 1, 1}, false, 0};
  MemRef<int8_t, 1> cache{std::vector<size_t>{CacheBytes}, false, 0};
};
using Forward = void (*)(Result *, MemRef<float, 1> *, MemRef<int8_t, 1> *,
                         MemRef<long long, 2> *, MemRef<long long, 2> *,
                         MemRef<int8_t, 1> *);
class Runner final : public buddy::runtime::InferenceRunner {
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
      throw std::runtime_error(dlerror());
    std::ifstream file;
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.open(manifest.weightPaths.at(0), std::ios::binary | std::ios::ate);
    const size_t floatBytes = file.tellg();
    MemRef<float, 1> parameters(
        std::vector<size_t>{floatBytes / sizeof(float)});
    file.seekg(0);
    file.read(reinterpret_cast<char *>(parameters.getData()), floatBytes);
    file.close();
    file.open(manifest.weightPaths.at(1), std::ios::binary | std::ios::ate);
    const size_t packedBytes = file.tellg();
    MemRef<int8_t, 1> packed(std::vector<size_t>{packedBytes});
    file.seekg(0);
    file.read(reinterpret_cast<char *>(packed.getData()), packedBytes);
    buddy::Text<long long, 2> prompt(config.prompt), generated;
    prompt.tokenizeQwen3(manifest.vocabPath, CacheCapacity);
    generated.loadVocab(manifest.vocabPath);
    const size_t count = prompt.getTokenCnt();
    if (!count || count >= CacheCapacity)
      throw std::runtime_error("Prompt leaves no room in the compiled context");
    MemRef<int8_t, 1> cache({CacheBytes}, 0);
    MemRef<long long, 2> input({1, 1}, 0), position({1, 1}, 0);
    buddy::Sampler sampler(config.samplerConfig);
    std::vector<int> tokens;
    int next = 0;
    const auto start = std::chrono::steady_clock::now();
    for (size_t step = 0; step + 1 < CacheCapacity; ++step) {
      input.getData()[0] = step < count ? prompt.getData()[step] : next;
      position.getData()[0] = step;
      Result result;
      forward(&result, &parameters, &packed, &input, &position, &cache);
      std::copy_n(result.cache.getData(), CacheBytes, cache.getData());
      if (step + 1 < count)
        continue;
      next = sampler.sample(result.logits.getData(),
                            result.logits.getSizes()[2], tokens);
      if (next == 151643 || next == 151645)
        break;
      tokens.push_back(next);
      generated.appendTokenIdx(next);
      if (config.maxNewTokens > 0 &&
          tokens.size() >= static_cast<size_t>(config.maxNewTokens))
        break;
    }
    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start;
    std::cout << generated.revertQwen3() << '\n';
    if (!config.suppressStats)
      std::cout << "Generated tokens: " << tokens.size()
                << ", inference time: " << elapsed.count() << "s\n";
  }
};
} // namespace
extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new Runner();
}
extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

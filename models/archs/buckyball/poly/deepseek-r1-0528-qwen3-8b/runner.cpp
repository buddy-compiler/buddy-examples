#include "SharedLibrary.h"
#include "testutils.h"
#include <buddy/LLM/TextContainer.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>
#include <buddy/runtime/llm/Sampler.h>
#include <resources.h>

#include <array>
#include <fstream>
#include <iostream>
#include <vector>

namespace {
constexpr size_t layers = 36, capacity = 256, vocabulary = 151936;
constexpr size_t cacheSize = 8 * capacity * 128;

class DeepSeekRunner final : public buddy::runtime::InferenceRunner {
public:
  void run(const buddy::runtime::RunConfig &config) override {
    const auto manifest =
        buddy::runtime::ModelManifest::loadFromRax(config.raxPath);
    auto library = model::openLibrary(manifest.soPath);
    auto forward = reinterpret_cast<int (*)(void **)>(
        dlsym(library.get(), manifest.codeObjects.at(0).entrySymbol.c_str()));
    if (!forward)
      throw std::runtime_error("Missing deepseek_run");
    resources::configure(
        std::stoull(manifest.moduleAttrs.at("image_base"), nullptr, 0));
    const auto &parametersPath = manifest.weightPaths.at(0);
    const auto &weightsPath = manifest.weightPaths.at(1);
    resources::Mapping parameters(
        parametersPath, resources::image.files.at(parametersPath).bytes, 64);
    resources::Mapping weights(
        weightsPath, resources::image.files.at(weightsPath).bytes, 64);
    std::ifstream orderFile(manifest.constants.at(2).path);
    std::vector<size_t> order;
    for (size_t index; orderFile >> index;)
      order.push_back(index);
    if (order.size() != 2 + 2 * layers || order.at(0) != 0)
      throw std::runtime_error("Invalid program input order");

    buddy::Text<long long, 2> prompt(config.prompt);
    const auto raw = prompt.byteTokenIds(manifest.vocabPath);
    std::vector<long long> ids(raw.begin(), raw.end());
    const size_t used = ids.size();
    if (!used || used >= capacity || config.maxNewTokens < 1)
      throw std::runtime_error("Invalid prompt or generation length");
    std::array<std::vector<float>, 2 * layers> cache, updated;
    for (size_t index = 0; index < cache.size(); ++index) {
      cache[index].resize(cacheSize);
      updated[index].resize(cacheSize);
    }
    std::vector<float> logits;
    auto step = [&](const long long *ids, size_t count, int64_t position) {
      logits.resize(count * vocabulary);
      std::array<void *, 2 + 2 * layers> inputs;
      inputs[0] = const_cast<long long *>(ids);
      inputs[1] = &position;
      for (size_t index = 0; index < cache.size(); ++index)
        inputs[2 + index] = cache[index].data();
      std::vector<void *> words{parameters.data(), weights.data()};
      for (size_t index : order) {
        words.push_back(inputs.at(index));
        if (index == 0)
          words.push_back(reinterpret_cast<void *>(count));
      }
      words.push_back(logits.data());
      words.push_back(reinterpret_cast<void *>(count));
      for (auto &buffer : updated)
        words.push_back(buffer.data());
      if (forward(words.data()))
        throw std::runtime_error("Device program failed");
      cache.swap(updated);
    };

    buddy::Sampler sampler(config.samplerConfig);
    std::vector<int> tokens;
    const auto begin = read_time_ns();
    step(ids.data(), used, 0);
    int next = sampler.sample(logits.data() + (used - 1) * vocabulary,
                              vocabulary, tokens);
    for (size_t position = used; position < capacity; ++position) {
      if (next == 151643 || next == 151645)
        break;
      tokens.push_back(next);
      if (tokens.size() == static_cast<size_t>(config.maxNewTokens))
        break;
      const long long id = next;
      step(&id, 1, position);
      next = sampler.sample(logits.data(), vocabulary, tokens);
    }
    const double elapsed = (read_time_ns() - begin) * 1e-9;
    std::cout << prompt.revertQwen3(tokens) << '\n';
    if (!config.suppressStats)
      std::cout << "Generated tokens: " << tokens.size()
                << ", inference time: " << elapsed << "s\n";
    if (!config.outputPath.empty()) {
      std::ofstream file;
      file.exceptions(std::ios::failbit | std::ios::badbit);
      file.open(config.outputPath, std::ios::binary);
      file.write(reinterpret_cast<const char *>(logits.data()),
                 logits.size() * sizeof(float));
    }
  }
};
} // namespace

extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new DeepSeekRunner();
}
extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

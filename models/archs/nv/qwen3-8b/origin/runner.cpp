#include "SharedLibrary.h"
#include "testutils.h"
#include <array>
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>
#include <buddy/runtime/llm/Sampler.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
namespace {
// The compiled graph: one decoder step of one token with a KV cache of 256
// positions in each of the 36 layers (8 KV heads of 128).
constexpr size_t layers = 36;
constexpr size_t capacity = 256;
constexpr size_t vocab = 151936;
constexpr size_t cacheSize = 8 * capacity * 128;

// An MLIR memref descriptor returned by the graph.
template <size_t N> struct Descriptor {
  float *allocated;
  float *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};

// The results of forward: the logits, then the keys of every layer, then the
// values of every layer.
struct Results {
  Descriptor<3> logits;
  Descriptor<4> keys[layers];
  Descriptor<4> values[layers];
};

// forward(params, tokens, start, keys0, values0, keys1, values1, ...): the
// graph takes the cache in the order Dynamo first uses it.
template <size_t> using Cache = MemRef<float, 4> *;
template <size_t... I>
void forward(void *function, Results *results, MemRef<float, 1> *parameters,
             MemRef<long long, 2> *tokens, MemRef<long long, 1> *start,
             std::array<MemRef<float, 4> *, 2 * layers> &cache,
             std::index_sequence<I...>) {
  using Forward =
      void (*)(Results *, MemRef<float, 1> *, MemRef<long long, 2> *,
               MemRef<long long, 1> *, Cache<I>...);
  reinterpret_cast<Forward>(function)(results, parameters, tokens, start,
                                      cache[I]...);
}

class QwenRunner final : public buddy::runtime::InferenceRunner {
public:
  void run(const buddy::runtime::RunConfig &config) override {
    const auto manifest =
        buddy::runtime::ModelManifest::loadFromRax(config.raxPath);
    std::vector<model::SharedLibrary> runtimes;
    for (const auto &path : manifest.dependentSoPaths)
      runtimes.push_back(model::openLibrary(path));
    auto library = model::openLibrary(manifest.soPath);
    void *function = dlsym(library.get(), "_mlir_ciface_forward");
    if (!function)
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

    buddy::Text<size_t, 2> prompt(config.prompt), generated;
    prompt.tokenizeQwen3(manifest.vocabPath, capacity);
    generated.loadVocab(manifest.vocabPath);
    const size_t used = prompt.getTokenCnt();
    if (used == 0 || used >= capacity)
      throw std::runtime_error("Prompt leaves no room in the KV cache");

    // The KV cache: keys and values of each layer, interleaved as the graph
    // takes them.
    std::vector<MemRef<float, 4>> buffers;
    for (size_t i = 0; i < 2 * layers; ++i)
      buffers.emplace_back(std::vector<size_t>{1, 8, capacity, 128}, 0.0f);
    std::array<MemRef<float, 4> *, 2 * layers> cache;
    for (size_t i = 0; i < 2 * layers; ++i)
      cache[i] = &buffers[i];
    MemRef<long long, 2> token({1, 1}, 0);
    MemRef<long long, 1> start({1}, 0);
    Results results;

    buddy::Sampler sampler(config.samplerConfig);
    std::vector<int> tokens;
    const auto begin = read_time_ns();
    // Feed the prompt one token at a time, then decode greedily until the
    // KV cache is full.
    for (size_t position = 0; position < capacity; ++position) {
      token.getData()[0] =
          position < used ? prompt.getData()[position] : tokens.back();
      start.getData()[0] = position;
      forward(function, &results, &parameters, &token, &start, cache,
              std::make_index_sequence<2 * layers>());
      for (size_t layer = 0; layer < layers; ++layer) {
        std::memcpy(buffers[2 * layer].getData(),
                    results.keys[layer].aligned + results.keys[layer].offset,
                    cacheSize * sizeof(float));
        std::memcpy(buffers[2 * layer + 1].getData(),
                    results.values[layer].aligned +
                        results.values[layer].offset,
                    cacheSize * sizeof(float));
        free(results.keys[layer].allocated);
        free(results.values[layer].allocated);
      }
      if (position + 1 < used) {
        free(results.logits.allocated);
        continue;
      }
      const int next = sampler.sample(
          results.logits.aligned + results.logits.offset, vocab, tokens);
      free(results.logits.allocated);
      if (next == 151643 || next == 151645)
        break;
      tokens.push_back(next);
      generated.appendTokenIdx(next);
      if (tokens.size() == static_cast<size_t>(config.maxNewTokens))
        break;
    }
    const double elapsed = (read_time_ns() - begin) * 1e-9;
    std::cout << generated.revertQwen3() << '\n';
    if (!config.suppressStats)
      std::cout << "Generated tokens: " << tokens.size()
                << ", inference time: " << elapsed << "s\n";
  }
};

} // namespace
extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new QwenRunner();
}
extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

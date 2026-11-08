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
// The prompt runs in chunks of this many tokens through forward_prefill.
constexpr size_t chunk = 64;
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

template <typename T> MemRef<T, 1> load(const std::string &path) {
  std::ifstream file;
  file.exceptions(std::ios::failbit | std::ios::badbit);
  file.open(path, std::ios::binary | std::ios::ate);
  const size_t bytes = file.tellg();
  if (bytes % sizeof(T))
    throw std::runtime_error("Invalid parameter size: " + path);
  MemRef<T, 1> values(std::vector<size_t>{bytes / sizeof(T)});
  file.seekg(0);
  file.read(reinterpret_cast<char *>(values.getData()), bytes);
  return values;
}

// forward(params, weights, tokens, start, keys0, values0, keys1, values1,
// ...): the FP32 parameters, the INT8 MXINT8 weights, then the cache in the
// order Dynamo first uses it.
template <size_t> using Cache = MemRef<float, 4> *;
template <size_t... I>
void forward(void *function, Results *results, MemRef<float, 1> *parameters,
             MemRef<int8_t, 1> *weights, MemRef<long long, 2> *tokens,
             MemRef<long long, 1> *start,
             std::array<MemRef<float, 4> *, 2 * layers> &cache,
             std::index_sequence<I...>) {
  using Forward =
      void (*)(Results *, MemRef<float, 1> *, MemRef<int8_t, 1> *,
               MemRef<long long, 2> *, MemRef<long long, 1> *, Cache<I>...);
  reinterpret_cast<Forward>(function)(results, parameters, weights, tokens,
                                      start, cache[I]...);
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
    void *prefill = dlsym(library.get(), "_mlir_ciface_forward_prefill");
    void *decode = dlsym(library.get(), "_mlir_ciface_forward_decode");
    if (!prefill || !decode)
      throw std::runtime_error("Missing forward_prefill or forward_decode: " +
                               std::string(dlerror()));
    auto parameters = load<float>(manifest.weightPaths.at(0));
    auto weights = load<int8_t>(manifest.weightPaths.at(1));

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
    MemRef<long long, 1> start({1}, 0);
    Results results;

    // Run `function` on `count` tokens written from `position` and copy the
    // updated KV cache back; the logits stay in `results`.
    auto step = [&](void *function, const long long *ids, size_t count,
                    size_t position) {
      MemRef<long long, 2> input({1, count}, 0);
      std::memcpy(input.getData(), ids, count * sizeof(long long));
      start.getData()[0] = position;
      forward(function, &results, &parameters, &weights, &input, &start, cache,
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
    };

    buddy::Sampler sampler(config.samplerConfig);
    std::vector<int> tokens;
    std::vector<long long> ids(prompt.getData(), prompt.getData() + used);
    const auto begin = read_time_ns();
    // Prefill in chunks (docs/ChunkedPrefill.md of Buddy-MLIR): the last
    // chunk ends at the last prompt token, and its rows the chunk before
    // already wrote get the same keys and values again. A shorter prompt is
    // padded with copies of its last token; the padding lies past the prompt,
    // is never attended to and is overwritten by the decode steps.
    size_t row;
    if (used >= chunk) {
      for (size_t position = 0; position + chunk < used; position += chunk)
        step(prefill, ids.data() + position, chunk, position);
      step(prefill, ids.data() + used - chunk, chunk, used - chunk);
      row = chunk - 1;
    } else {
      ids.resize(chunk, ids.back());
      step(prefill, ids.data(), chunk, 0);
      row = used - 1;
    }
    int next = sampler.sample(results.logits.aligned + results.logits.offset +
                                  row * vocab,
                              vocab, tokens);
    free(results.logits.allocated);
    // Decode greedily one token at a time until the KV cache is full.
    for (size_t position = used; position < capacity; ++position) {
      if (next == 151643 || next == 151645)
        break;
      tokens.push_back(next);
      generated.appendTokenIdx(next);
      if (tokens.size() == static_cast<size_t>(config.maxNewTokens))
        break;
      const long long id = next;
      step(decode, &id, 1, position);
      next = sampler.sample(results.logits.aligned + results.logits.offset,
                            vocab, tokens);
      free(results.logits.allocated);
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

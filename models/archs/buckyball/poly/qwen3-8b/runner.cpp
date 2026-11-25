#include "SharedLibrary.h"
#include "testutils.h"
#include <buddy/LLM/TextContainer.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>
#include <buddy/runtime/llm/Sampler.h>
#include <resources.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

namespace {
constexpr size_t capacity = 256;

class QwenRunner final : public buddy::runtime::InferenceRunner {
public:
  void run(const buddy::runtime::RunConfig &config) override {
    std::cerr << "Loading model: " << config.raxPath << std::endl;
    const auto manifest =
        buddy::runtime::ModelManifest::loadFromRax(config.raxPath);
    auto library = model::openLibrary(manifest.soPath);
    auto forward = reinterpret_cast<int (*)(void **)>(
        dlsym(library.get(), manifest.codeObjects.at(0).entrySymbol.c_str()));
    if (!forward)
      throw std::runtime_error("Missing qwen_run");
    resources::configure(
        std::stoull(manifest.moduleAttrs.at("image_base"), nullptr, 0));
    const auto &parametersPath = manifest.weightPaths.at(0);
    const auto &weightsPath = manifest.weightPaths.at(1);
    const auto &tablePath = manifest.weightPaths.at(2);
    std::cerr << "Mapping weights from DDR" << std::endl;
    resources::Mapping parameters(
        parametersPath, resources::image.files.at(parametersPath).bytes, 64);
    resources::Mapping weights(
        weightsPath, resources::image.files.at(weightsPath).bytes, 64);
    resources::Mapping table(tablePath,
                             resources::image.files.at(tablePath).bytes, 64);
    std::cerr << "Weights ready: "
              << resources::image.files.at(parametersPath).bytes +
                     resources::image.files.at(weightsPath).bytes +
                     resources::image.files.at(tablePath).bytes
              << " bytes mapped" << std::endl;
    std::ifstream orderFile(manifest.constants.at(3).path);
    std::vector<size_t> order;
    for (size_t index; orderFile >> index;)
      order.push_back(index);
    // Each row describes one explicit state plane: heads, row bytes,
    // code bytes and the byte filling the remaining exponent positions.
    struct StateLayout {
      size_t heads, rowBytes, codeBytes, exponent;
    };
    std::ifstream layoutFile(manifest.constants.at(5).path);
    std::vector<StateLayout> layout;
    StateLayout plane;
    while (layoutFile >> plane.heads >> plane.rowBytes >> plane.codeBytes >>
           plane.exponent)
      layout.push_back(plane);
    if (!layoutFile.eof() || layout.empty() ||
        order.size() != 2 + layout.size() || order.at(0) != 0)
      throw std::runtime_error("Invalid program state layout");

    // The prompt: its token ids, of the baremetal program (prompt/ids.data)
    // for the check against the reference, else of the given text. The check
    // decodes the eight steps of baremetal.cpp.
    const bool check = !config.referencePath.empty();
    buddy::Text<long long, 2> prompt(config.prompt);
    std::vector<long long> ids;
    if (check) {
      std::ifstream file;
      file.exceptions(std::ios::failbit | std::ios::badbit);
      file.open(config.inputPath, std::ios::binary | std::ios::ate);
      ids.resize(static_cast<size_t>(file.tellg()) / sizeof(long long));
      file.seekg(0);
      file.read(reinterpret_cast<char *>(ids.data()),
                ids.size() * sizeof(long long));
    } else {
      const auto raw = prompt.byteTokenIds(manifest.vocabPath);
      ids.assign(raw.begin(), raw.end());
    }
    const size_t used = ids.size();
    const size_t generate = check ? 8 : config.maxNewTokens;
    if (!used || used >= capacity || generate < 1)
      throw std::runtime_error("Invalid prompt or generation length");
    const size_t requested = used + generate;
    if (requested > capacity)
      throw std::runtime_error("Request exceeds KV cache capacity");
    size_t cacheCapacity = (used + 31) / 32 * 32;
    std::vector<std::vector<uint8_t>> cache(layout.size()),
        updated(layout.size());
    auto initialize = [&](std::vector<uint8_t> &buffer, size_t index,
                          size_t rows) {
      const auto &plane = layout.at(index);
      buffer.assign(plane.heads * rows * plane.rowBytes, 0);
      for (size_t row = 0; row < plane.heads * rows; ++row)
        std::fill(buffer.begin() + row * plane.rowBytes + plane.codeBytes,
                  buffer.begin() + (row + 1) * plane.rowBytes, plane.exponent);
    };
    for (size_t index = 0; index < cache.size(); ++index) {
      initialize(cache[index], index, cacheCapacity);
      initialize(updated[index], index, cacheCapacity);
    }
    // The candidates of the next token the program returns: the largest
    // logits of the last position on each rank's piece of the vocabulary
    // and their indices, the ranks in order. They hold the `candidates`
    // largest logits of the vocabulary: the sampler reads no other. Greedy
    // decoding picks the first largest, the first maximum of the logits.
    // candidates.txt: the candidates of a rank, then their number.
    std::ifstream candidatesFile(manifest.constants.at(4).path);
    size_t candidates = 0, total = 0;
    if (!(candidatesFile >> candidates >> total))
      throw std::runtime_error("Invalid candidate counts");
    const auto &sampling = config.samplerConfig;
    if (!check && sampling.temperature != 0.0f &&
        (sampling.topK < 1 || static_cast<size_t>(sampling.topK) > candidates))
      throw std::runtime_error("Sampling needs a top-k of at most " +
                               std::to_string(candidates));
    if (!check && sampling.repeatPenalty != 1.0f)
      throw std::runtime_error("No repetition penalty over the candidates");
    std::vector<float> values(total);
    std::vector<int64_t> indices(total);
    auto step = [&](const long long *ids, size_t count, int64_t position) {
      const size_t needed =
          (static_cast<size_t>(position) + count + 31) / 32 * 32;
      if (needed > cacheCapacity) {
        for (size_t index = 0; index < cache.size(); ++index) {
          const auto &plane = layout[index];
          std::vector<uint8_t> grown;
          initialize(grown, index, needed);
          for (size_t head = 0; head < plane.heads; ++head)
            std::copy_n(cache[index].data() +
                            head * cacheCapacity * plane.rowBytes,
                        cacheCapacity * plane.rowBytes,
                        grown.data() + head * needed * plane.rowBytes);
          cache[index].swap(grown);
          initialize(updated[index], index, needed);
        }
        cacheCapacity = needed;
      }
      std::vector<void *> inputs(2 + layout.size());
      inputs[0] = const_cast<long long *>(ids);
      inputs[1] = &position;
      for (size_t index = 0; index < cache.size(); ++index)
        inputs[2 + index] = cache[index].data();
      std::vector<void *> words{parameters.data(), weights.data(),
                                table.data()};
      for (size_t index : order) {
        words.push_back(inputs.at(index));
        if (index == 0)
          words.push_back(reinterpret_cast<void *>(count));
        else if (index >= 2)
          words.push_back(reinterpret_cast<void *>(cacheCapacity));
      }
      words.push_back(values.data());
      words.push_back(indices.data());
      for (auto &buffer : updated) {
        words.push_back(buffer.data());
        words.push_back(reinterpret_cast<void *>(cacheCapacity));
      }
      if (forward(words.data()))
        throw std::runtime_error("Device program failed");
      cache.swap(updated);
    };

    // The eight greedy steps of baremetal.cpp against the reference of
    // buddy-pytorch-golden.py (prompt/reference.data: the candidates of every
    // step, then the token it picks): the same lines and bounds, and a
    // failure when the candidates or a token differ.
    if (check) {
      std::vector<float> expected(8 * total);
      std::vector<int64_t> goldTokens(8);
      std::ifstream golden;
      golden.exceptions(std::ios::failbit | std::ios::badbit);
      golden.open(config.referencePath, std::ios::binary);
      golden.read(reinterpret_cast<char *>(expected.data()),
                  expected.size() * sizeof(float));
      golden.read(reinterpret_cast<char *>(goldTokens.data()),
                  goldTokens.size() * sizeof(int64_t));
      std::vector<float> stepCandidates(8 * total);
      unsigned different = 0;
      long long position = 0, next = 0;
      const uint64_t first = read_counter();
      for (unsigned index = 0; index < 8; ++index) {
        if (index == 0)
          step(ids.data(), used, 0);
        else
          step(&next, 1, position);
        position += index == 0 ? used : 1;
        size_t best = 0;
        for (size_t i = 1; i < total; ++i)
          if (values[i] > values[best])
            best = i;
        next = indices[best];
        std::copy_n(values.data(), total,
                    stepCandidates.data() + index * total);
        different += next != goldTokens[index];
        printf("Token %u: %lld, reference %lld\n", index + 1, next,
               static_cast<long long>(goldTokens[index]));
      }
      const uint64_t cycles = read_counter() - first;
      printf("Generated tokens: 8, %u differ from the reference\n", different);
      printf("Inference cycles: %lu\n", static_cast<unsigned long>(cycles));
      if (check_output(stepCandidates.data(), expected.data(), 8 * total,
                       1e-2f) ||
          different)
        throw std::runtime_error("Output differs from the reference");
      return;
    }

    buddy::Sampler sampler(sampling);
    std::vector<int> tokens;
    auto sample = [&] {
      return static_cast<int>(
          indices[sampler.sample(values.data(), values.size(), {})]);
    };
    std::cerr << "Prefill: " << used
              << " tokens, generation limit: " << config.maxNewTokens
              << std::endl;
    const auto begin = read_time_ns();
    step(ids.data(), used, 0);
    int token = sample();
    for (size_t position = used; position < capacity; ++position) {
      if (token == 151643 || token == 151645)
        break;
      tokens.push_back(token);
      std::cerr << "Token " << tokens.size() << "/" << config.maxNewTokens
                << ": " << token << std::endl;
      if (tokens.size() == static_cast<size_t>(config.maxNewTokens))
        break;
      const long long id = token;
      step(&id, 1, position);
      token = sample();
    }
    const double elapsed = (read_time_ns() - begin) * 1e-9;
    std::cout << prompt.revertQwen3(tokens) << std::endl;
    if (!config.suppressStats)
      std::cout << "Generated tokens: " << tokens.size()
                << ", inference time: " << elapsed << "s\n";
    if (!config.outputPath.empty()) {
      std::ofstream file;
      file.exceptions(std::ios::failbit | std::ios::badbit);
      file.open(config.outputPath, std::ios::binary);
      file.write(reinterpret_cast<const char *>(values.data()),
                 values.size() * sizeof(float));
      file.write(reinterpret_cast<const char *>(indices.data()),
                 indices.size() * sizeof(int64_t));
    }
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

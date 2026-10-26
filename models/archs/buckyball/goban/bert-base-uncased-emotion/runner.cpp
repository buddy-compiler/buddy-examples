#include "SharedLibrary.h"
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using model::openLibrary;
using model::SharedLibrary;

constexpr size_t sequenceLength = 32;
constexpr size_t classes = 6;

using Forward = void (*)(MemRef<float, 2> *, MemRef<float, 1> *,
                         MemRef<int64_t, 2> *, MemRef<int64_t, 2> *,
                         MemRef<int64_t, 2> *, MemRef<int64_t, 2> *);

class BertRunner final : public buddy::runtime::InferenceRunner {
public:
  void run(const buddy::runtime::RunConfig &config) override {
    const auto manifest =
        buddy::runtime::ModelManifest::loadFromRax(config.raxPath);
    std::vector<SharedLibrary> runtimes;
    for (const auto &path : manifest.dependentSoPaths)
      runtimes.push_back(openLibrary(path));
    auto library = openLibrary(manifest.soPath);
    const auto &entry = manifest.codeObjects.at(0).entrySymbol;
    auto forward =
        reinterpret_cast<Forward>(dlsym(library.get(), entry.c_str()));
    if (!forward)
      throw std::runtime_error("Missing " + entry + ": " +
                               std::string(dlerror()));

    std::ifstream weights;
    weights.exceptions(std::ios::failbit | std::ios::badbit);
    weights.open(manifest.weightPaths.at(0), std::ios::binary | std::ios::ate);
    const size_t bytes = weights.tellg();
    if (bytes % sizeof(float))
      throw std::runtime_error("Invalid FP32 parameter byte count");
    MemRef<float, 1> parameters(std::vector<size_t>{bytes / sizeof(float)});
    weights.seekg(0);
    weights.read(reinterpret_cast<char *>(parameters.getData()), bytes);

    // The prompt padded to the sequence length; the mask hides the padding.
    buddy::Text<int64_t, 2> ids(config.prompt);
    ids.tokenizeBert(manifest.vocabPath, sequenceLength, true, true);
    MemRef<int64_t, 2> types({1, sequenceLength}, 0),
        mask({1, sequenceLength}, 0), positions({1, sequenceLength}, 0);
    for (size_t i = 0; i < sequenceLength; ++i) {
      positions.getData()[i] = i;
      mask.getData()[i] = i < ids.getTokenCnt();
    }

    // The compiled function returns an owned memref descriptor.
    MemRef<float, 2> output(std::vector<size_t>{1, classes}, false, 0);
    const auto start = std::chrono::steady_clock::now();
    forward(&output, &parameters, &ids, &types, &mask, &positions);
    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start;
    const float *logits = output.getData();
    const auto prediction = std::max_element(logits, logits + classes) - logits;
    double sum = 0.0;
    for (size_t i = 0; i < classes; ++i)
      sum += std::exp(logits[i] - logits[prediction]);
    constexpr const char *labels[classes] = {"sadness", "joy",  "love",
                                             "anger",   "fear", "surprise"};
    std::cout << "Classification: " << labels[prediction] << '\n'
              << "Probability: " << 1.0 / sum << '\n';
    if (!config.suppressStats)
      std::cout << "Inference time: " << elapsed.count() << "s\n";
  }
};

} // namespace

extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new BertRunner();
}

extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

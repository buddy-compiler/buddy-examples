#include "SharedLibrary.h"
#include "testutils.h"
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>

#include <algorithm>
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
                         MemRef<int32_t, 1> *, MemRef<int8_t, 1> *,
                         MemRef<int64_t, 2> *, MemRef<int64_t, 2> *,
                         MemRef<int64_t, 2> *, MemRef<int64_t, 2> *);

template <typename T> MemRef<T, 1> load(const std::string &path) {
  std::ifstream file;
  file.exceptions(std::ios::failbit | std::ios::badbit);
  file.open(path, std::ios::binary | std::ios::ate);
  const size_t bytes = file.tellg();
  if (bytes % sizeof(T))
    throw std::runtime_error("Invalid parameter byte count: " + path);
  MemRef<T, 1> values(std::vector<size_t>{bytes / sizeof(T)});
  file.seekg(0);
  file.read(reinterpret_cast<char *>(values.getData()), bytes);
  return values;
}

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

    auto parameters = load<float>(manifest.weightPaths.at(0));
    auto bias = load<int32_t>(manifest.weightPaths.at(1));
    auto weights = load<int8_t>(manifest.weightPaths.at(2));

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
    const auto start = read_time_ns();
    forward(&output, &parameters, &bias, &weights, &ids, &types, &mask,
            &positions);
    const double elapsed = (read_time_ns() - start) * 1e-9;
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
      std::cout << "Inference time: " << elapsed << "s\n";
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

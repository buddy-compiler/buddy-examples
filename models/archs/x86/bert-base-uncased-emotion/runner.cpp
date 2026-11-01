#include "SharedLibrary.h"
#include "testutils.h"
#include <algorithm>
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
namespace {
using Forward = void (*)(MemRef<float, 2> *, MemRef<float, 1> *,
                         MemRef<int8_t, 1> *, MemRef<long long, 2> *,
                         MemRef<long long, 2> *, MemRef<long long, 2> *,
                         MemRef<long long, 2> *);

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

class BertRunner final : public buddy::runtime::InferenceRunner {
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
    // The parameters grouped by type: FP32, then the INT8 MXINT8 weights.
    auto parameters = load<float>(manifest.weightPaths.at(0));
    auto weights = load<int8_t>(manifest.weightPaths.at(1));
    buddy::Text<long long, 2> input(config.prompt);
    input.tokenizeBert(manifest.vocabPath, 32, true, true);
    MemRef<long long, 2> types({1, 32}, 0), mask({1, 32}, 0),
        positions({1, 32}, 0);
    for (size_t i = 0; i < 32; ++i) {
      positions.getData()[i] = i;
      mask.getData()[i] = i < input.getTokenCnt();
    }
    MemRef<float, 2> output(std::vector<size_t>{1, 6}, false, 0);
    const auto start = read_time_ns();
    // The imported ABI orders inputs as IDs, positions, types, then mask.
    forward(&output, &parameters, &weights, &input, &positions, &types, &mask);
    const double elapsed = (read_time_ns() - start) * 1e-9;
    const float *logits = output.getData();
    const size_t prediction = std::max_element(logits, logits + 6) - logits;
    constexpr const char *labels[] = {"sadness", "joy",  "love",
                                      "anger",   "fear", "surprise"};
    double sum = 0;
    for (size_t i = 0; i < 6; ++i)
      sum += std::exp(logits[i] - logits[prediction]);
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

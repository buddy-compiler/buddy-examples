#include "SharedLibrary.h"
#include <algorithm>
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
namespace {
using Forward = void (*)(MemRef<float, 2> *, MemRef<float, 1> *,
                         MemRef<long long, 2> *, MemRef<long long, 2> *,
                         MemRef<long long, 2> *, MemRef<long long, 2> *);
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
    std::ifstream weights;
    weights.exceptions(std::ios::failbit | std::ios::badbit);
    weights.open(manifest.weightPaths.at(0), std::ios::binary | std::ios::ate);
    const size_t bytes = weights.tellg();
    if (bytes % sizeof(float))
      throw std::runtime_error("Invalid FP32 parameter size");
    MemRef<float, 1> parameters(std::vector<size_t>{bytes / sizeof(float)});
    weights.seekg(0);
    weights.read(reinterpret_cast<char *>(parameters.getData()), bytes);
    buddy::Text<long long, 2> input(config.prompt);
    input.tokenizeBert(manifest.vocabPath, 32, true, true);
    MemRef<long long, 2> types({1, 32}, 0), mask({1, 32}, 0),
        positions({1, 32}, 0);
    for (size_t i = 0; i < 32; ++i) {
      positions.getData()[i] = i;
      mask.getData()[i] = i < input.getTokenCnt();
    }
    MemRef<float, 2> output(std::vector<size_t>{1, 6}, false, 0);
    const auto start = std::chrono::steady_clock::now();
    // The imported ABI orders inputs as IDs, positions, types, then mask.
    forward(&output, &parameters, &input, &positions, &types, &mask);
    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start;
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

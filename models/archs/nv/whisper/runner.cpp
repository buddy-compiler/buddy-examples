#include "SharedLibrary.h"
#include "testutils.h"
#include <algorithm>
#include <buddy/Core/Container.h>
#include <buddy/DAP/AudioContainer.h>
#include <buddy/DAP/DAP.h>
#include <buddy/LLM/TextContainer.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
namespace {
using Forward = void (*)(MemRef<float, 3> *, MemRef<float, 1> *,
                         MemRef<int8_t, 1> *, MemRef<float, 3> *,
                         MemRef<size_t, 2> *);
template <typename T> MemRef<T, 1> loadParameters(const std::string &path) {
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
class WhisperRunner final : public buddy::runtime::InferenceRunner {
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
    // The parameters grouped by type: FP32, then the INT8 weights.
    auto parameters = loadParameters<float>(manifest.weightPaths.at(0));
    auto weights = loadParameters<int8_t>(manifest.weightPaths.at(1));
    dap::Audio<double, 1> audio(config.audioPath);
    MemRef<float, 3> features({1, 80, 3000});
    dap::whisperPreprocess(&audio, &features);
    MemRef<size_t, 2> tokens({1, 448}, 50258);
    buddy::Text<size_t, 2> generated;
    generated.loadVocab(manifest.vocabPath);
    const auto start = read_time_ns();
    size_t count = 0;
    while (count < 447 && (config.maxNewTokens == 0 ||
                           count < static_cast<size_t>(config.maxNewTokens))) {
      MemRef<float, 3> logits(std::vector<size_t>{1, 448, 51865}, false, 0);
      forward(&logits, &parameters, &weights, &features, &tokens);
      const float *row = logits.getData() + count * 51865;
      const size_t token = std::max_element(row, row + 51865) - row;
      if (token == 50257)
        break;
      tokens.getData()[++count] = token;
      generated.appendTokenIdx(token);
    }
    std::cout << generated.revertWhisper() << '\n';
    if (!config.suppressStats) {
      const double elapsed = (read_time_ns() - start) * 1e-9;
      std::cout << "Generated tokens: " << count
                << ", inference time: " << elapsed << "s\n";
    }
  }
};
} // namespace
extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new WhisperRunner();
}
extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

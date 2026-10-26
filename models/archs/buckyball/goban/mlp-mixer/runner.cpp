#include "SharedLibrary.h"
#include <buddy/Core/Container.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using model::openLibrary;
using model::SharedLibrary;

constexpr size_t patches = 196;
constexpr size_t patchSize = 768;
constexpr size_t classes = 1000;

using Forward = void (*)(MemRef<float, 2> *, MemRef<float, 1> *,
                         MemRef<float, 2> *);

MemRef<float, 1> load(const std::string &path) {
  std::ifstream file;
  file.exceptions(std::ios::failbit | std::ios::badbit);
  file.open(path, std::ios::binary | std::ios::ate);
  const size_t bytes = file.tellg();
  if (bytes % sizeof(float))
    throw std::runtime_error("Invalid FP32 byte count: " + path);
  MemRef<float, 1> values(std::vector<size_t>{bytes / sizeof(float)});
  file.seekg(0);
  file.read(reinterpret_cast<char *>(values.getData()), bytes);
  return values;
}

class MixerRunner final : public buddy::runtime::InferenceRunner {
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

    auto parameters = load(manifest.weightPaths.at(0));
    // The input is the prepared [196, 768] FP32 patches (output/input.f32).
    auto values = load(config.imagePath);
    if (values.getSize() != patches * patchSize)
      throw std::runtime_error("Mixer requires 196x768 patches");
    MemRef<float, 2> input(std::vector<size_t>{patches, patchSize});
    std::copy_n(values.getData(), patches * patchSize, input.getData());

    // The compiled function returns an owned memref descriptor.
    MemRef<float, 2> output(std::vector<size_t>{1, classes}, false, 0);
    const auto start = std::chrono::steady_clock::now();
    forward(&output, &parameters, &input);
    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start;
    const float *logits = output.getData();
    const auto prediction = std::max_element(logits, logits + classes) - logits;
    double sum = 0.0;
    for (size_t i = 0; i < classes; ++i)
      sum += std::exp(logits[i] - logits[prediction]);
    std::cout << "Classification Index: " << prediction << '\n'
              << "Probability: " << 1.0 / sum << '\n';
    if (!config.suppressStats)
      std::cout << "Inference time: " << elapsed.count() << "s\n";
  }
};

} // namespace

extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new MixerRunner();
}

extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

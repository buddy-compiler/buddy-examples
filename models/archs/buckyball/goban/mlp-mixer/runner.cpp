#include "SharedLibrary.h"
#include "model.h"
#include "testutils.h"
#include <buddy/Core/Container.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

using mixer::Forward;

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

    auto parameters = load<float>(manifest.weightPaths.at(0));
    auto bias = load<int32_t>(manifest.weightPaths.at(1));
    auto weights = load<int8_t>(manifest.weightPaths.at(2));
    // The input is the prepared [196, 768] FP32 patches (output/input.f32).
    auto values = load<float>(config.imagePath);
    if (values.getSize() != patches * patchSize)
      throw std::runtime_error("Mixer requires 196x768 patches");
    MemRef<float, 2> input(std::vector<size_t>{patches, patchSize});
    std::copy_n(values.getData(), patches * patchSize, input.getData());

    // The compiled function returns an owned memref descriptor.
    mixer::Parameters f32{parameters.getData(),
                          parameters.getData(),
                          0,
                          {int64_t(parameters.getSize())},
                          {1}};
    mixer::Bias i32{
        bias.getData(), bias.getData(), 0, {int64_t(bias.getSize())}, {1}};
    mixer::Weights i8{weights.getData(),
                      weights.getData(),
                      0,
                      {int64_t(weights.getSize())},
                      {1}};
    mixer::Input patchesInput{input.getData(),
                              input.getData(),
                              0,
                              {patches, patchSize},
                              {patchSize, 1}};
    mixer::Output output{};
    const auto start = read_time_ns();
    forward(&output, &f32, &i32, &i8, &patchesInput);
    const double elapsed = (read_time_ns() - start) * 1e-9;
    const float *logits = output.data();
    const auto prediction = std::max_element(logits, logits + classes) - logits;
    double sum = 0.0;
    for (size_t i = 0; i < classes; ++i)
      sum += std::exp(logits[i] - logits[prediction]);
    std::cout << "Classification Index: " << prediction << '\n'
              << "Probability: " << 1.0 / sum << '\n';
    if (!config.outputPath.empty()) {
      std::ofstream dump;
      dump.exceptions(std::ios::failbit | std::ios::badbit);
      dump.open(config.outputPath, std::ios::binary);
      dump.write(reinterpret_cast<const char *>(logits),
                 classes * sizeof(float));
      dump.close();
    }
    std::free(output.allocated);
    if (!config.suppressStats)
      std::cout << "Inference time: " << elapsed << "s\n";
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

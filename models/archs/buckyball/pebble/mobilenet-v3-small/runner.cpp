#include "SharedLibrary.h"
#include <buddy/Core/Container.h>
#include <buddy/DIP/DIP.h>
#include <buddy/DIP/ImgContainer.h>
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

using Forward = void (*)(MemRef<float, 2> *, MemRef<float, 1> *,
                         MemRef<int32_t, 1> *, MemRef<int8_t, 1> *,
                         MemRef<float, 4> *);

template <typename T> MemRef<T, 1> loadParameters(const std::string &path) {
  std::ifstream file;
  file.exceptions(std::ios::failbit | std::ios::badbit);
  file.open(path, std::ios::binary | std::ios::ate);
  const size_t bytes = file.tellg();
  if (bytes % sizeof(T))
    throw std::runtime_error("Invalid parameter byte count: " + path);
  MemRef<T, 1> parameters(std::vector<size_t>{bytes / sizeof(T)});
  file.seekg(0);
  file.read(reinterpret_cast<char *>(parameters.getData()), bytes);
  return parameters;
}

std::string label(const std::string &path, size_t index) {
  std::ifstream file;
  file.exceptions(std::ios::failbit | std::ios::badbit);
  file.open(path);
  std::string line;
  for (size_t i = 0; i <= index; ++i)
    std::getline(file, line);
  return line;
}

class MobileNetRunner final : public buddy::runtime::InferenceRunner {
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
    auto parameters = loadParameters<float>(manifest.weightPaths.at(0));
    auto quantizedBias = loadParameters<int32_t>(manifest.weightPaths.at(1));
    auto quantizedWeights = loadParameters<int8_t>(manifest.weightPaths.at(2));

    // Resize to 224x224, normalize with the ImageNet statistics, and pass the
    // image in NHWC, the layout of the compiled feature extractor.
    dip::Image<float, 4> image(config.imagePath, dip::DIP_RGB, true /* norm */);
    MemRef<float, 4> resized = dip::Resize4D_NCHW(
        &image, dip::INTERPOLATION_TYPE::BILINEAR_INTERPOLATION,
        {1, 3, 224, 224});
    constexpr float Mean[] = {0.485f, 0.456f, 0.406f};
    constexpr float Std[] = {0.229f, 0.224f, 0.225f};
    MemRef<float, 4> input(std::vector<size_t>{1, 224, 224, 3});
    const float *pixels = resized.getData();
    for (size_t channel = 0; channel < 3; ++channel)
      for (size_t pixel = 0; pixel < 224 * 224; ++pixel)
        input.getData()[pixel * 3 + channel] =
            (pixels[channel * 224 * 224 + pixel] - Mean[channel]) /
            Std[channel];

    // The compiled function returns an owned memref descriptor.
    MemRef<float, 2> output(std::vector<size_t>{1, 1000}, false, 0);
    const auto start = std::chrono::steady_clock::now();
    forward(&output, &parameters, &quantizedBias, &quantizedWeights, &input);
    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start;
    const float *logits = output.getData();
    const auto prediction = std::max_element(logits, logits + 1000) - logits;
    double sum = 0.0;
    for (size_t i = 0; i < 1000; ++i)
      sum += std::exp(logits[i] - logits[prediction]);
    std::cout << "Classification Index: " << prediction << '\n'
              << "Classification: " << label(manifest.vocabPath, prediction)
              << '\n'
              << "Probability: " << 1.0 / sum << '\n';
    if (!config.suppressStats)
      std::cout << "Inference time: " << elapsed.count() << "s\n";
  }
};

} // namespace

extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new MobileNetRunner();
}

extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

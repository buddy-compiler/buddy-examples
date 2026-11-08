#include "ImageClassifier.h"
#include "SharedLibrary.h"
#include "testutils.h"
#include <buddy/Core/Container.h>
#include <buddy/DIP/ImgContainer.h>
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

using model::Forward;

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

class AlexNetRunner final : public buddy::runtime::InferenceRunner {
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
    auto quantizedBias = load<int32_t>(manifest.weightPaths.at(1));
    auto quantizedWeights = load<int8_t>(manifest.weightPaths.at(2));

    // The model reads NHWC images normalized with the ImageNet statistics.
    dip::Image<float, 4> image(config.imagePath, dip::DIP_RGB, true);
    if (image.getWidth() != 224 || image.getHeight() != 224)
      throw std::runtime_error("AlexNet requires a 224x224 image");
    constexpr float mean[] = {0.485f, 0.456f, 0.406f};
    constexpr float deviation[] = {0.229f, 0.224f, 0.225f};
    MemRef<float, 4> input(std::vector<size_t>{1, 224, 224, 3});
    for (size_t channel = 0; channel < 3; ++channel)
      for (size_t pixel = 0; pixel < 224 * 224; ++pixel)
        input.getData()[pixel * 3 + channel] =
            (image.getData()[channel * 224 * 224 + pixel] - mean[channel]) /
            deviation[channel];

    // The compiled function returns an owned memref descriptor.
    model::Parameters f32{parameters.getData(),
                          parameters.getData(),
                          0,
                          {int64_t(parameters.getSize())},
                          {1}};
    model::Bias i32{quantizedBias.getData(),
                    quantizedBias.getData(),
                    0,
                    {int64_t(quantizedBias.getSize())},
                    {1}};
    model::Weights i8{quantizedWeights.getData(),
                      quantizedWeights.getData(),
                      0,
                      {int64_t(quantizedWeights.getSize())},
                      {1}};
    model::Image imageInput{input.getData(),
                            input.getData(),
                            0,
                            {1, 224, 224, 3},
                            {150528, 672, 3, 1}};
    model::Scores output{};
    const auto start = read_time_ns();
    forward(&output, &f32, &i32, &i8, &imageInput);
    const double elapsed = (read_time_ns() - start) * 1e-9;
    const float *logits = output.data();
    const size_t prediction = std::max_element(logits, logits + 1000) - logits;
    double sum = 0.0;
    for (size_t i = 0; i < 1000; ++i)
      sum += std::exp(logits[i] - logits[prediction]);
    std::cout << "Classification Index: " << prediction << '\n'
              << "Probability: " << 1.0 / sum << '\n';
    if (!config.outputPath.empty()) {
      std::ofstream dump;
      dump.exceptions(std::ios::failbit | std::ios::badbit);
      dump.open(config.outputPath, std::ios::binary);
      dump.write(reinterpret_cast<const char *>(logits), 1000 * sizeof(float));
      dump.close();
    }
    std::free(output.allocated);
    if (!config.suppressStats)
      std::cout << "Inference time: " << elapsed << "s\n";
  }
};

} // namespace

extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new AlexNetRunner();
}

extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

#include "SharedLibrary.h"
#include <algorithm>
#include <buddy/Core/Container.h>
#include <buddy/DIP/ImgContainer.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
namespace {
using Forward = void (*)(MemRef<float, 2> *, MemRef<float, 1> *,
                         MemRef<float, 4> *);
class AlexNetRunner final : public buddy::runtime::InferenceRunner {
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
    dip::Image<float, 4> image(config.imagePath, dip::DIP_RGB, true);
    if (image.getWidth() != 224 || image.getHeight() != 224)
      throw std::runtime_error("AlexNet requires a 224x224 image");
    MemRef<float, 4> input(std::vector<size_t>{1, 3, 224, 224});
    std::copy_n(image.getData(), 3 * 224 * 224, input.getData());
    constexpr float mean[] = {0.485f, 0.456f, 0.406f};
    constexpr float deviation[] = {0.229f, 0.224f, 0.225f};
    for (size_t channel = 0; channel < 3; ++channel)
      for (size_t pixel = 0; pixel < 224 * 224; ++pixel) {
        const size_t i = channel * 224 * 224 + pixel;
        input.getData()[i] =
            (input.getData()[i] - mean[channel]) / deviation[channel];
      }
    MemRef<float, 2> output(std::vector<size_t>{1, 1000}, false, 0);
    const auto start = std::chrono::steady_clock::now();
    forward(&output, &parameters, &input);
    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start;
    const float *logits = output.getData();
    const size_t prediction = std::max_element(logits, logits + 1000) - logits;
    double sum = 0;
    for (size_t i = 0; i < 1000; ++i)
      sum += std::exp(logits[i] - logits[prediction]);
    std::cout << "Classification Index: " << prediction << '\n'
              << "Probability: " << 1.0 / sum << '\n';
    if (!config.suppressStats)
      std::cout << "Inference time: " << elapsed.count() << "s\n";
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

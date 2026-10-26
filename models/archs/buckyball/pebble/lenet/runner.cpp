#include "SharedLibrary.h"
#include <buddy/Core/Container.h>
#include <buddy/DIP/ImgContainer.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using model::openLibrary;
using model::SharedLibrary;

uint32_t readBE32(std::ifstream &stream) {
  std::array<uint8_t, 4> data;
  stream.read(reinterpret_cast<char *>(data.data()), data.size());
  return (uint32_t(data[0]) << 24) | (uint32_t(data[1]) << 16) |
         (uint32_t(data[2]) << 8) | uint32_t(data[3]);
}

using Forward = void (*)(MemRef<float, 2> *, MemRef<float, 1> *,
                         MemRef<int32_t, 1> *, MemRef<int8_t, 1> *,
                         MemRef<float, 4> *);

void evaluateMnist(const buddy::runtime::RunConfig &config, Forward forward,
                   MemRef<float, 1> &parameters,
                   MemRef<int32_t, 1> &quantizedBias,
                   MemRef<int8_t, 1> &quantizedWeights) {
  const auto directory =
      std::filesystem::path(config.datasetPath) / "MNIST/raw";
  std::ifstream images, labels;
  images.exceptions(std::ios::failbit | std::ios::badbit);
  labels.exceptions(std::ios::failbit | std::ios::badbit);
  images.open(directory / "t10k-images-idx3-ubyte", std::ios::binary);
  labels.open(directory / "t10k-labels-idx1-ubyte", std::ios::binary);
  if (readBE32(images) != 2051 || readBE32(labels) != 2049)
    throw std::runtime_error("Invalid MNIST file magic");
  const size_t count = readBE32(images);
  if (!count || readBE32(labels) != count || readBE32(images) != 28 ||
      readBE32(images) != 28)
    throw std::runtime_error("Invalid MNIST count or image dimensions");
  std::vector<uint8_t> pixels(count * 28 * 28), expected(count);
  images.read(reinterpret_cast<char *>(pixels.data()), pixels.size());
  labels.read(reinterpret_cast<char *>(expected.data()), expected.size());

  MemRef<float, 4> input(std::vector<size_t>{1, 28, 28, 1});
  size_t correct = 0;
  const auto start = std::chrono::steady_clock::now();
  for (size_t sample = 0; sample < count; ++sample) {
    for (size_t i = 0; i < 28 * 28; ++i)
      input.getData()[i] =
          (pixels[sample * 28 * 28 + i] / 255.0f) * 2.0f - 1.0f;
    // Each invocation returns an owned output descriptor, released here.
    MemRef<float, 2> output(std::vector<size_t>{1, 10}, false, 0);
    forward(&output, &parameters, &quantizedBias, &quantizedWeights, &input);
    const float *logits = output.getData();
    correct +=
        std::max_element(logits, logits + 10) - logits == expected[sample];
  }
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count();
  std::cout << "top1=" << correct << '/' << count << '\n';
  if (!config.suppressStats)
    std::cout << "@BBPERF {\"model\":\"lenet\",\"scope\":\"model_execution\","
                 "\"samples\":"
              << count << ",\"correct\":" << correct
              << ",\"accuracy_metric\":\"top1\",\"unit\":\"images/s\","
                 "\"elapsed_ns\":"
              << elapsed << "}\n";
}

void classifyImage(const buddy::runtime::RunConfig &config, Forward forward,
                   MemRef<float, 1> &parameters,
                   MemRef<int32_t, 1> &quantizedBias,
                   MemRef<int8_t, 1> &quantizedWeights) {
  // Preserve floating-point grayscale conversion; DIP_GRAYSCALE truncates it.
  dip::Image<float, 4> rgb(config.imagePath, dip::DIP_RGB);
  if (rgb.getWidth() != 28 || rgb.getHeight() != 28)
    throw std::runtime_error("LeNet requires a 28x28 image");
  MemRef<float, 4> input(std::vector<size_t>{1, 28, 28, 1});
  const float *pixels = rgb.getData();
  for (size_t i = 0; i < 28 * 28; ++i) {
    float gray = (0.114f * pixels[2 * 28 * 28 + i] +
                  0.587f * pixels[28 * 28 + i] + 0.299f * pixels[i]) /
                 255.0f;
    input.getData()[i] = gray * 2.0f - 1.0f;
  }

  // The compiled function returns an owned memref descriptor.
  MemRef<float, 2> output(std::vector<size_t>{1, 10}, false, 0);
  const auto start = std::chrono::steady_clock::now();
  forward(&output, &parameters, &quantizedBias, &quantizedWeights, &input);
  const std::chrono::duration<double> elapsed =
      std::chrono::steady_clock::now() - start;
  const float *logits = output.getData();
  const auto prediction = std::max_element(logits, logits + 10) - logits;
  double sum = 0.0;
  for (size_t i = 0; i < 10; ++i)
    sum += std::exp(logits[i] - logits[prediction]);
  std::cout << "Classification: " << prediction << '\n'
            << "Probability: " << 1.0 / sum << '\n';
  if (!config.suppressStats)
    std::cout << "Inference time: " << elapsed.count() << "s\n";
}

class LeNetRunner final : public buddy::runtime::InferenceRunner {
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

    weights.close();
    weights.open(manifest.weightPaths.at(1), std::ios::binary | std::ios::ate);
    const size_t biasBytes = weights.tellg();
    if (biasBytes % sizeof(int32_t))
      throw std::runtime_error("Invalid INT32 bias byte count");
    MemRef<int32_t, 1> quantizedBias(
        std::vector<size_t>{biasBytes / sizeof(int32_t)});
    weights.seekg(0);
    weights.read(reinterpret_cast<char *>(quantizedBias.getData()), biasBytes);

    weights.close();
    weights.open(manifest.weightPaths.at(2), std::ios::binary | std::ios::ate);
    const size_t weightBytes = weights.tellg();
    MemRef<int8_t, 1> quantizedWeights(std::vector<size_t>{weightBytes});
    weights.seekg(0);
    weights.read(reinterpret_cast<char *>(quantizedWeights.getData()),
                 weightBytes);

    if (!config.datasetPath.empty())
      evaluateMnist(config, forward, parameters, quantizedBias,
                    quantizedWeights);
    else
      classifyImage(config, forward, parameters, quantizedBias,
                    quantizedWeights);
  }
};

} // namespace

extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new LeNetRunner();
}

extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

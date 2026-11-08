#include "testutils.h"
#include <buddy/Core/Container.h>
#include <buddy/DIP/ImgContainer.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

uint32_t readBE32(std::ifstream &stream) {
  std::array<uint8_t, 4> data;
  stream.read(reinterpret_cast<char *>(data.data()), data.size());
  return (uint32_t(data[0]) << 24) | (uint32_t(data[1]) << 16) |
         (uint32_t(data[2]) << 8) | uint32_t(data[3]);
}

} // namespace

// The parameters are grouped by type: FP32, INT32 biases, INT8 weights.
extern "C" void _mlir_ciface_forward(MemRef<float, 2> *output,
                                     MemRef<float, 1> *parameters,
                                     MemRef<int32_t, 1> *quantizedBias,
                                     MemRef<int8_t, 1> *quantizedWeights,
                                     MemRef<float, 4> *input);

namespace {

struct Parameters {
  MemRef<float, 1> &values;
  MemRef<int32_t, 1> &bias;
  MemRef<int8_t, 1> &weights;
};

void forward(MemRef<float, 2> *output, Parameters &parameters,
             MemRef<float, 4> *input) {
  _mlir_ciface_forward(output, &parameters.values, &parameters.bias,
                       &parameters.weights, input);
}

template <typename T> MemRef<T, 1> load(const std::string &path) {
  std::ifstream file;
  file.exceptions(std::ios::failbit | std::ios::badbit);
  file.open(path, std::ios::binary | std::ios::ate);
  const size_t bytes = file.tellg();
  if (bytes % sizeof(T))
    throw std::runtime_error("Invalid parameter byte count: " + path);
  MemRef<T, 1> tensor(std::vector<size_t>{bytes / sizeof(T)});
  file.seekg(0);
  file.read(reinterpret_cast<char *>(tensor.getData()), bytes);
  return tensor;
}

void evaluateMnist(const buddy::runtime::RunConfig &config,
                   Parameters &parameters) {
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

  MemRef<float, 4> input(std::vector<size_t>{1, 1, 28, 28});
  size_t correct = 0;
  const auto start = read_time_ns();
  for (size_t sample = 0; sample < count; ++sample) {
    for (size_t i = 0; i < 28 * 28; ++i)
      input.getData()[i] =
          (pixels[sample * 28 * 28 + i] / 255.0f) * 2.0f - 1.0f;
    // Each invocation returns an owned output descriptor, released here.
    MemRef<float, 2> output(std::vector<size_t>{1, 10}, false, 0);
    forward(&output, parameters, &input);
    const float *logits = output.getData();
    correct +=
        std::max_element(logits, logits + 10) - logits == expected[sample];
  }
  const auto elapsed = (read_time_ns() - start);
  std::cout << "top1=" << correct << '/' << count << '\n';
  if (!config.suppressStats)
    std::cout << "@BBPERF {\"model\":\"lenet\",\"scope\":\"model_execution\","
                 "\"samples\":"
              << count << ",\"correct\":" << correct
              << ",\"accuracy_metric\":\"top1\",\"unit\":\"images/s\","
                 "\"elapsed_ns\":"
              << elapsed << "}\n";
}

void classifyImage(const buddy::runtime::RunConfig &config,
                   Parameters &parameters) {
  // Preserve floating-point grayscale conversion; DIP_GRAYSCALE truncates it.
  dip::Image<float, 4> rgb(config.imagePath, dip::DIP_RGB);
  if (rgb.getWidth() != 28 || rgb.getHeight() != 28)
    throw std::runtime_error("LeNet requires a 28x28 image");
  MemRef<float, 4> input(std::vector<size_t>{1, 1, 28, 28});
  const float *pixels = rgb.getData();
  for (size_t i = 0; i < 28 * 28; ++i) {
    float gray = (0.114f * pixels[2 * 28 * 28 + i] +
                  0.587f * pixels[28 * 28 + i] + 0.299f * pixels[i]) /
                 255.0f;
    input.getData()[i] = gray * 2.0f - 1.0f;
  }

  // The compiled function returns an owned memref descriptor.
  MemRef<float, 2> output(std::vector<size_t>{1, 10}, false, 0);
  const auto start = read_time_ns();
  forward(&output, parameters, &input);
  const double elapsed = (read_time_ns() - start) * 1e-9;
  const float *logits = output.getData();
  const auto prediction = std::max_element(logits, logits + 10) - logits;
  double sum = 0.0;
  for (size_t i = 0; i < 10; ++i)
    sum += std::exp(logits[i] - logits[prediction]);
  std::cout << "Classification: " << prediction << '\n'
            << "Probability: " << 1.0 / sum << '\n';
  if (!config.suppressStats)
    std::cout << "Inference time: " << elapsed << "s\n";
}

class LeNetRunner final : public buddy::runtime::InferenceRunner {
public:
  void run(const buddy::runtime::RunConfig &config) override {
    const auto manifest =
        buddy::runtime::ModelManifest::loadFromRax(config.raxPath);
    auto values = load<float>(manifest.weightPaths.at(0));
    auto bias = load<int32_t>(manifest.weightPaths.at(1));
    auto weights = load<int8_t>(manifest.weightPaths.at(2));
    Parameters parameters{values, bias, weights};

    if (!config.datasetPath.empty())
      evaluateMnist(config, parameters);
    else
      classifyImage(config, parameters);
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

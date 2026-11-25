// LeNet compiled for Gemmini, as one program for spike with the Gemmini
// extension: the first MNIST test images against the quantized model in
// software (buddy-pytorch-calibrate.py).
#include <buddy/Core/Container.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// The parameters are grouped by type: FP32, INT32 biases, INT8 weights.
extern "C" void _mlir_ciface_forward(MemRef<float, 2> *output,
                                     MemRef<float, 1> *parameters,
                                     MemRef<int32_t, 1> *quantizedBias,
                                     MemRef<int8_t, 1> *quantizedWeights,
                                     MemRef<float, 4> *input);

template <typename T> std::vector<T> load(const std::string &path) {
  std::ifstream file;
  file.exceptions(std::ios::failbit | std::ios::badbit);
  file.open(path, std::ios::binary | std::ios::ate);
  std::vector<T> values(size_t(file.tellg()) / sizeof(T));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(values.data()), values.size() * sizeof(T));
  return values;
}

template <typename T> MemRef<T, 1> tensor(const std::vector<T> &values) {
  intptr_t sizes[1] = {intptr_t(values.size())};
  return MemRef<T, 1>(values.data(), sizes);
}

int main(int, char **argv) {
  const std::string directory = argv[1];
  const auto values = load<float>(directory + "/arg0.data");
  const auto bias = load<int32_t>(directory + "/arg1.data");
  const auto weights = load<int8_t>(directory + "/arg2.data");
  const auto images = load<float>(directory + "/check-images.data");
  const auto logits = load<float>(directory + "/check-logits.data");
  auto parameters = tensor(values);
  auto quantizedBias = tensor(bias);
  auto quantizedWeights = tensor(weights);

  const size_t count = images.size() / (28 * 28);
  MemRef<float, 4> input(std::vector<size_t>{1, 1, 28, 28});
  size_t same = 0;
  float largest = 0;
  for (size_t sample = 0; sample < count; ++sample) {
    std::memcpy(input.getData(), images.data() + sample * 28 * 28,
                28 * 28 * sizeof(float));
    // Each invocation returns an owned output descriptor.
    MemRef<float, 2> output(std::vector<size_t>{1, 10}, false, 0);
    _mlir_ciface_forward(&output, &parameters, &quantizedBias,
                         &quantizedWeights, &input);
    const float *got = output.getData();
    const float *expected = logits.data() + sample * 10;
    same += std::max_element(got, got + 10) - got ==
            std::max_element(expected, expected + 10) - expected;
    for (size_t i = 0; i < 10; ++i)
      largest = std::max(largest, std::fabs(got[i] - expected[i]));
  }
  std::cout << "lenet on gemmini: " << same << '/' << count
            << " predictions as the quantized model, largest logit difference "
            << largest << '\n';
  return same == count ? 0 : 1;
}

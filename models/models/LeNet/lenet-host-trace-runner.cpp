//===- lenet-host-trace-runner.cpp ----------------------------------------===//

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <buddy/Core/Container.h>

constexpr size_t ParamsSize = 44426;

extern "C" void _mlir_ciface_subgraph0(MemRef<float, 2> *output,
                                       MemRef<float, 4> *input,
                                       MemRef<float, 4> *arg1,
                                       MemRef<float, 1> *arg2,
                                       MemRef<float, 4> *arg3,
                                       MemRef<float, 1> *arg4,
                                       MemRef<float, 2> *arg5,
                                       MemRef<float, 1> *arg6,
                                       MemRef<float, 2> *arg7,
                                       MemRef<float, 1> *arg8,
                                       MemRef<float, 2> *arg9,
                                       MemRef<float, 1> *arg10);

static void loadBinaryFile(const std::string &path, float *dst, size_t count) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open file: " + path);
  }
  file.read(reinterpret_cast<char *>(dst), sizeof(float) * count);
  if (file.fail()) {
    throw std::runtime_error("failed to read file: " + path);
  }
}

static void softmax(float *input, size_t size) {
  float maxValue = *std::max_element(input, input + size);
  double sum = 0.0;
  for (size_t i = 0; i < size; ++i) {
    sum += std::exp(input[i] - maxValue);
  }
  for (size_t i = 0; i < size; ++i) {
    input[i] = std::exp(input[i] - maxValue) / sum;
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <arg0.data> <input_nchw.bin>\n";
    return 1;
  }

  MemRef<float, 1> params({ParamsSize});
  loadBinaryFile(argv[1], params.getData(), ParamsSize);

  MemRef<float, 4> input({1, 1, 28, 28});
  loadBinaryFile(argv[2], input.getData(), 28 * 28);

  intptr_t conv1WeightShape[4] = {6, 1, 5, 5};
  intptr_t conv1BiasShape[1] = {6};
  intptr_t conv2WeightShape[4] = {16, 6, 5, 5};
  intptr_t conv2BiasShape[1] = {16};
  intptr_t fc1WeightShape[2] = {120, 256};
  intptr_t fc1BiasShape[1] = {120};
  intptr_t fc2WeightShape[2] = {84, 120};
  intptr_t fc2BiasShape[1] = {84};
  intptr_t fc3WeightShape[2] = {10, 84};
  intptr_t fc3BiasShape[1] = {10};

  MemRef<float, 4> conv1Weight(params.getData(), conv1WeightShape);
  MemRef<float, 1> conv1Bias(params.getData() + 150, conv1BiasShape);
  MemRef<float, 4> conv2Weight(params.getData() + 156, conv2WeightShape);
  MemRef<float, 1> conv2Bias(params.getData() + 2556, conv2BiasShape);
  MemRef<float, 2> fc1Weight(params.getData() + 2572, fc1WeightShape);
  MemRef<float, 1> fc1Bias(params.getData() + 33292, fc1BiasShape);
  MemRef<float, 2> fc2Weight(params.getData() + 33412, fc2WeightShape);
  MemRef<float, 1> fc2Bias(params.getData() + 43492, fc2BiasShape);
  MemRef<float, 2> fc3Weight(params.getData() + 43576, fc3WeightShape);
  MemRef<float, 1> fc3Bias(params.getData() + 44416, fc3BiasShape);
  MemRef<float, 2> output({1, 10});

  _mlir_ciface_subgraph0(&output, &input, &conv1Weight, &conv1Bias,
                         &conv2Weight, &conv2Bias, &fc1Weight, &fc1Bias,
                         &fc2Weight, &fc2Bias, &fc3Weight, &fc3Bias);

  softmax(output.getData(), 10);
  int maxIdx = 0;
  float maxVal = output.getData()[0];
  for (int i = 1; i < 10; ++i) {
    if (output.getData()[i] > maxVal) {
      maxVal = output.getData()[i];
      maxIdx = i;
    }
  }
  std::cout << "Classification: " << maxIdx << "\n";
  std::cout << "Probability: " << maxVal << "\n";
  return 0;
}

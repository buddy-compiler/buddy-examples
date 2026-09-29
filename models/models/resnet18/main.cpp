//===- buddy-resnet-main.cpp ----------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "testutils.h"
#include <buddy/Core/Container.h>
#include <buddy/DIP/DIP.h>
#include <buddy/DIP/ImgContainer.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>


// Declare the resnet C interface.
extern "C" void _mlir_ciface_forward(MemRef<float, 2> *output,
                                     MemRef<float, 1> *arg0,
                                     MemRef<int8_t, 1> *weights,
                                     MemRef<float, 4> *input);



template <typename T>
MemRef<T, 1> loadBinary(const std::string &path) {
  std::cout << "\033[34;1m[Log] \033[0mLoading " << path << std::endl;
  const auto loadStart = std::chrono::steady_clock::now();
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  input.exceptions(std::ios::failbit | std::ios::badbit);
  const size_t bytes = input.tellg();
  if (bytes % sizeof(T))
    throw std::runtime_error("invalid tensor byte count: " + path);
  MemRef<T, 1> tensor(std::vector<size_t>{bytes / sizeof(T)});
  input.seekg(0);
  input.read(reinterpret_cast<char *>(tensor.getData()), bytes);
  const std::chrono::duration<double> loadTime =
      std::chrono::steady_clock::now() - loadStart;
  std::cout << "\033[34;1m[Log] \033[0mLoad time: " << loadTime.count()
            << "s" << std::endl;
  return tensor;
}

// Softmax function.
void softmax(float *input, size_t size) {
  size_t i;
  float max_value = -INFINITY;
  double sum = 0.0;
  // Find the maximum value in the input array for numerical stability.
  for (i = 0; i < size; ++i) {
    if (max_value < input[i]) {
      max_value = input[i];
    }
  }
  // Calculate the sum of the exponentials of the input elements, normalized by
  // the max value.
  for (i = 0; i < size; ++i) {
    sum += exp(input[i] - max_value);
  }
  // Normalize the input array with the softmax calculation.
  for (i = 0; i < size; ++i) {
    input[i] = exp(input[i] - max_value) / sum;
  }
}

std::string getLabel(int idx) {
  std::string resnetDir = "./";
  std::ifstream in(resnetDir + "/Labels.txt");
  assert(in.is_open() && "Could not read the label file.");
  std::string label;
  for (int i = 0; i < idx; ++i)
    std::getline(in, label);
  std::getline(in, label);
  in.close();
  return label;
}

int main(int argc, char **argv) {
  if (argc != 2) throw std::runtime_error("expected an image path");
  // Print the title of this example.
  const std::string title = "ResNet Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  // Define the sizes of the input and output tensors.
  intptr_t sizesOutput[2] = {1, 1000};

  // Create input and output containers for the image and model output.
  std::string resnetDir = "./";
  std::string imgPath = argv[1];
  dip::Image<float, 4> input(imgPath, dip::DIP_RGB, true /* norm */);
  MemRef<float, 4> inputResize = dip::Resize4D_NCHW(
      &input, dip::INTERPOLATION_TYPE::BILINEAR_INTERPOLATION,
      {1, 3, 224, 224} /*{image_cols, image_rows}*/);
  constexpr float Mean[] = {0.485f, 0.456f, 0.406f};
  constexpr float Std[] = {0.229f, 0.224f, 0.225f};
  float *inputData = inputResize.getData();
  for (size_t channel = 0; channel < 3; ++channel) {
    for (size_t pixel = 0; pixel < 224 * 224; ++pixel) {
      size_t index = channel * 224 * 224 + pixel;
      inputData[index] = (inputData[index] - Mean[channel]) / Std[channel];
    }
  }

  MemRef<float, 2> output(sizesOutput);

  auto paramsContainer = loadBinary<float>(resnetDir + "/resnet18.payload/params.f32");
  auto weightsContainer = loadBinary<int8_t>(resnetDir + "/resnet18.payload/weights.bin");

  std::cout << "\033[34;1m[Log] \033[0mStarting inference..." << std::endl;
  unsigned long start = read_cycles();
  _mlir_ciface_forward(&output, &paramsContainer, &weightsContainer, &inputResize);
  unsigned long end = read_cycles();
  std::cout << "Cycle count: " << end - start << std::endl;

  auto out = output.getData();
  softmax(out, 1000);
  // Find the classification and print the result.
  float maxVal = out[0];
  int maxIdx = 0;
  for (int i = 1; i < 1000; ++i) {
    if (out[i] > maxVal) {
      maxVal = out[i];
      maxIdx = i;
    }
  }
  std::cout << "Classification Index: " << maxIdx << std::endl;
  std::cout << "Classification: " << getLabel(maxIdx) << std::endl;
  std::cout << "Probability: " << maxVal << std::endl;

  return 0;
}

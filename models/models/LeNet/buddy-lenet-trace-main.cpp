//===- buddy-lenet-trace-main.cpp -----------------------------------------===//
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

#include "include/testutils.h"
#include <buddy/Core/Container.h>
#include <buddy/DIP/ImgContainer.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

constexpr size_t ParamsSize = 44426;
const std::string ImgName = "8.bmp";

extern "C" void _mlir_ciface_forward(MemRef<float, 2> *output,
                                     MemRef<float, 1> *arg0,
                                     dip::Image<float, 4> *input);
extern "C" void buddyTraceTensorF32(const char *tag, const char *layout,
                                    int64_t rank, const int64_t *shape,
                                    const float *data, int64_t elemCount);

static void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

static void loadParameters(const std::string &paramFilePath,
                           MemRef<float, 1> &params) {
  const auto loadStart = std::chrono::high_resolution_clock::now();
  std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
  if (!paramFile.is_open()) {
    throw std::runtime_error("[Error] Failed to open params file!");
  }
  printLogLabel();
  std::cout << "Loading params..." << std::endl;
  printLogLabel();
  std::cout << "Params file: " << std::filesystem::canonical(paramFilePath)
            << std::endl;
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * (params.getSize()));
  if (paramFile.fail()) {
    throw std::runtime_error("Error occurred while reading params file!");
  }
  paramFile.close();
  const auto loadEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> loadTime =
      loadEnd - loadStart;
  printLogLabel();
  std::cout << "Params load time: " << (double)(loadTime.count()) / 1000
            << "s\n"
            << std::endl;
}

static void softmax(float *input, size_t size) {
  size_t i;
  float maxValue = -INFINITY;
  double sum = 0.0;
  for (i = 0; i < size; ++i) {
    if (maxValue < input[i]) {
      maxValue = input[i];
    }
  }
  for (i = 0; i < size; ++i) {
    sum += exp(input[i] - maxValue);
  }
  for (i = 0; i < size; ++i) {
    input[i] = exp(input[i] - maxValue) / sum;
  }
}

static void traceInput(dip::Image<float, 4> &input) {
  int64_t shape[4] = {1, 1, 28, 28};
  buddyTraceTensorF32("input_nchw", "nchw", 4, shape, input.getData(), 784);
}

static void normalizeLeNetInput(dip::Image<float, 4> &input) {
  float *data = input.getData();
  const size_t elemCount = input.getSize();
  for (size_t i = 0; i < elemCount; ++i) {
    data[i] = data[i] * 2.0f - 1.0f;
  }
}

int main() {
  const std::string title = "LeNet Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  intptr_t sizesOutput[2] = {1, 10};
  std::string lenetDir = "./";
  std::string imgPath = lenetDir + "/images/" + ImgName;
  dip::Image<float, 4> input(imgPath, dip::DIP_GRAYSCALE, true /* norm */);
  normalizeLeNetInput(input);
  traceInput(input);

  MemRef<float, 2> output(sizesOutput);

  std::string paramsDir = lenetDir + "/arg0.data";
  MemRef<float, 1> paramsContainer({ParamsSize});
  loadParameters(paramsDir, paramsContainer);

  unsigned long start = read_cycles();
  _mlir_ciface_forward(&output, &paramsContainer, &input);
  unsigned long end = read_cycles();

  auto out = output.getData();
  softmax(out, 10);
  printLogLabel();
  std::cout << "Inference Cycles taken: " << end - start << std::endl;
  std::cout << std::endl;

  float maxVal = 0;
  float maxIdx = 0;
  for (int i = 0; i < 10; ++i) {
    if (out[i] > maxVal) {
      maxVal = out[i];
      maxIdx = i;
    }
  }

  std::cout << "Results: " << std::endl;
  std::cout << "Classification: " << maxIdx << std::endl;
  std::cout << "Probability: " << maxVal << std::endl;
  return 0;
}

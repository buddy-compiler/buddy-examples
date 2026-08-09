//===- buddy-lenet-main.cpp -----------------------------------------------===//
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
#include <buddy/DIP/ImgContainer.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

constexpr size_t ParamsSize = 44426;
constexpr size_t MnistCount = 10000;
constexpr size_t MnistPixels = 28 * 28;
const std::string ImgName = "8.bmp";

struct Opts {
  std::string weights = "./arg0.data";
  std::string dataset;
};

static Opts parseArgs(int argc, char **argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--weights") {
      if (++i >= argc)
        throw std::runtime_error("--weights needs a path");
      o.weights = argv[i];
    } else if (a == "--dataset") {
      if (++i >= argc)
        throw std::runtime_error("--dataset needs a path");
      o.dataset = argv[i];
    } else {
      throw std::runtime_error("unknown arg: " + a);
    }
  }
  return o;
}

static uint32_t readBe32(std::ifstream &f) {
  uint8_t b[4];
  f.read(reinterpret_cast<char *>(b), 4);
  if (f.fail())
    throw std::runtime_error("failed to read u32");
  return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) |
         (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

static std::vector<uint8_t> loadMnistImages(const std::string &root) {
  std::string path = root + "/MNIST/raw/t10k-images-idx3-ubyte";
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("failed to open MNIST images: " + path);
  if (readBe32(f) != 2051)
    throw std::runtime_error("bad MNIST image magic: " + path);
  if (readBe32(f) != MnistCount)
    throw std::runtime_error("bad MNIST image count: " + path);
  if (readBe32(f) != 28 || readBe32(f) != 28)
    throw std::runtime_error("bad MNIST image shape: " + path);
  std::vector<uint8_t> data(MnistCount * MnistPixels);
  f.read(reinterpret_cast<char *>(data.data()), data.size());
  if (f.gcount() != static_cast<std::streamsize>(data.size()))
    throw std::runtime_error("short MNIST image read: " + path);
  return data;
}

static std::vector<uint8_t> loadMnistLabels(const std::string &root) {
  std::string path = root + "/MNIST/raw/t10k-labels-idx1-ubyte";
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("failed to open MNIST labels: " + path);
  if (readBe32(f) != 2049)
    throw std::runtime_error("bad MNIST label magic: " + path);
  if (readBe32(f) != MnistCount)
    throw std::runtime_error("bad MNIST label count: " + path);
  std::vector<uint8_t> data(MnistCount);
  f.read(reinterpret_cast<char *>(data.data()), data.size());
  if (f.gcount() != static_cast<std::streamsize>(data.size()))
    throw std::runtime_error("short MNIST label read: " + path);
  return data;
}

static int argmax(const float *data, size_t n) {
  int best = 0;
  float bestVal = data[0];
  for (size_t i = 1; i < n; ++i) {
    if (data[i] > bestVal) {
      bestVal = data[i];
      best = static_cast<int>(i);
    }
  }
  return best;
}

/// Declare LeNet forward function.
extern "C" void _mlir_ciface_forward(MemRef<float, 2> *output,
                                     MemRef<float, 1> *arg0,
                                     dip::Image<float, 4> *input);

/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

static void validateParamsFile(const std::string &path) {
  if (!std::filesystem::exists(path))
    throw std::runtime_error("params file not found: " + path);
  auto bytes = std::filesystem::file_size(path);
  if (bytes != ParamsSize * sizeof(float))
    throw std::runtime_error("params file size mismatch: " + path);
}

/// Load parameters into data container.
void loadParameters(const std::string &paramFilePath,
                    MemRef<float, 1> &params) {
  validateParamsFile(paramFilePath);
  if (params.getSize() != ParamsSize)
    throw std::runtime_error("params container size mismatch");

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
                 sizeof(float) * params.getSize());
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

/// Softmax function to convert logits to probabilities.
void softmax(float *input, size_t size) {
  size_t i;
  float max_value = -INFINITY;
  double sum = 0.0;
  for (i = 0; i < size; ++i) {
    if (max_value < input[i]) {
      max_value = input[i];
    }
  }
  for (i = 0; i < size; ++i) {
    sum += exp(input[i] - max_value);
  }
  for (i = 0; i < size; ++i) {
    input[i] = exp(input[i] - max_value) / sum;
  }
}

static void normalizeLeNetInput(dip::Image<float, 4> &input) {
  float *data = input.getData();
  const size_t elemCount = input.getSize();
  for (size_t i = 0; i < elemCount; ++i) {
    data[i] = data[i] * 2.0f - 1.0f;
  }
}

static void fillMnistImage(float *dst, const uint8_t *src) {
  for (size_t p = 0; p < MnistPixels; ++p)
    dst[p] = (src[p] / 255.0f) * 2.0f - 1.0f;
}

int main(int argc, char **argv) {
  const std::string title = "LeNet Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  Opts opts = parseArgs(argc, argv);
  intptr_t sizesOutput[2] = {1, 10};
  MemRef<float, 1> paramsContainer({ParamsSize});
  loadParameters(opts.weights, paramsContainer);

  if (!opts.dataset.empty()) {
    auto images = loadMnistImages(opts.dataset);
    auto labels = loadMnistLabels(opts.dataset);
    size_t correct = 0;
    std::vector<float> buf(MnistPixels);
    intptr_t inSizes[4] = {1, 1, 28, 28};
    MemRef<float, 2> output(sizesOutput);
    for (size_t i = 0; i < MnistCount; ++i) {
      fillMnistImage(buf.data(), images.data() + i * MnistPixels);
      dip::Image<float, 4> input(buf.data(), inSizes);
      _mlir_ciface_forward(&output, &paramsContainer, &input);
      if (argmax(output.getData(), 10) == labels[i])
        ++correct;
    }
    std::cout << "top1=" << correct << "/10000" << std::endl;
    return 0;
  }

  std::string imgPath = "./images/" + ImgName;
  dip::Image<float, 4> input(imgPath, dip::DIP_GRAYSCALE, true /* norm */);
  normalizeLeNetInput(input);
  MemRef<float, 2> output(sizesOutput);

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

  // images/8.bmp must classify as digit 8; wrong label means DMA/CoW or model bug.
  constexpr int expect = 8;
  if (static_cast<int>(maxIdx) != expect) {
    std::cerr << "FAIL expected classification " << expect << ", got " << maxIdx
              << std::endl;
    return 1;
  }
  std::cout << "PASS classification=" << maxIdx << std::endl;
  return 0;
}

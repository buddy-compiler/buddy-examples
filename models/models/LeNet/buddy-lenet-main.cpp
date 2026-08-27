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
#include <bbhw/isa/isa.h>
#include <buddy/Core/Container.h>
#include <buddy/DIP/ImgContainer.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <unistd.h>
#include <vector>

constexpr size_t ParamsSize = 236;
constexpr size_t WeightsSize = 44190;
constexpr size_t ScalesSize = 1088;
constexpr size_t MnistCount = 10000;
constexpr size_t MnistPixels = 28 * 28;
const std::string ImgName = "8.bmp";


struct Opts {
  std::string params = "./lenet.payload/params.f32";
  std::string weights = "./lenet.payload/weights.i8";
  std::string scales = "./lenet.payload/scales.bin";
  std::string dataset;
};

static Opts parseArgs(int argc, char **argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--params") {
      if (++i >= argc)
        throw std::runtime_error("--params needs a path");
      o.params = argv[i];
    } else if (a == "--weights") {
      if (++i >= argc)
        throw std::runtime_error("--weights needs a path");
      o.weights = argv[i];
    } else if (a == "--scales") {
      if (++i >= argc)
        throw std::runtime_error("--scales needs a path");
      o.scales = argv[i];
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
                                     MemRef<float, 1> *params,
                                     MemRef<int8_t, 1> *weights,
                                     MemRef<float, 4> *input);

class BorrowedImage : public MemRef<float, 4> {
public:
  BorrowedImage(float *data, intptr_t sizes[4])
      : MemRef<float, 4>(sizes, false, 0) {
    allocated = aligned = data;
  }
  ~BorrowedImage() { allocated = aligned = nullptr; }
};

template <typename T, size_t N>
class BorrowedBuffer : public MemRef<T, N> {
public:
  BorrowedBuffer(T *data, intptr_t sizes[N]) : MemRef<T, N>(sizes, false, 0) {
    this->allocated = this->aligned = data;
  }
  ~BorrowedBuffer() { this->allocated = this->aligned = nullptr; }
};

/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

template <typename T>
void loadBinary(const std::string &path, T *data, size_t count) {
  const auto loadStart = std::chrono::high_resolution_clock::now();
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0)
    throw std::runtime_error("failed to open binary file: " + path);
  printLogLabel();
  std::cout << "Loading " << path << std::endl;
  const size_t bytes = sizeof(T) * count;
  if (read(fd, data, bytes) != static_cast<ssize_t>(bytes))
    throw std::runtime_error("short binary file: " + path);
  close(fd);
  const auto loadEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> loadTime =
      loadEnd - loadStart;
  printLogLabel();
  std::cout << "Load time: " << (double)(loadTime.count()) / 1000 << "s\n";
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

static uint32_t le32(const uint8_t *p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
         (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

static void loadLeNetInput(float *out) {
  uint8_t bmp[3190];
  int fd = open("./images/8.bmp", O_RDONLY);
  if (fd < 0 || read(fd, bmp, sizeof(bmp)) != sizeof(bmp))
    throw std::runtime_error("failed to read images/8.bmp");
  close(fd);
  if (le32(bmp + 10) != 54 || le32(bmp + 18) != 28 ||
      le32(bmp + 22) != 28 || bmp[28] != 32)
    throw std::runtime_error("invalid LeNet BMP");
  for (size_t y = 0; y < 28; ++y)
    for (size_t x = 0; x < 28; ++x) {
      const uint8_t *pixel = bmp + 54 + ((27 - y) * 28 + x) * 4;
      float gray = (0.114f * pixel[0] + 0.587f * pixel[1] + 0.299f * pixel[2]) / 255.0f;
      out[y * 28 + x] = gray * 2.0f - 1.0f;
    }
}

static void normalizeLeNetInput(MemRef<float, 4> &input) {
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
  static float paramsData[ParamsSize] __attribute__((aligned(64)));
  static int8_t weightsData[WeightsSize] __attribute__((aligned(64)));
  static uint8_t scalesData[ScalesSize] __attribute__((aligned(64)));
  intptr_t paramsSize[1] = {ParamsSize};
  intptr_t weightsSize[1] = {WeightsSize};
  BorrowedBuffer<float, 1> paramsContainer(paramsData, paramsSize);
  BorrowedBuffer<int8_t, 1> weightsContainer(weightsData, weightsSize);
  loadBinary(opts.params, paramsData, ParamsSize);
  loadBinary(opts.weights, weightsData, WeightsSize);
  loadBinary(opts.scales, scalesData, ScalesSize);
  bb_mvin_mmio(reinterpret_cast<uintptr_t>(scalesData), 16,
               ScalesSize / 16, 16);

  if (!opts.dataset.empty()) {
    auto images = loadMnistImages(opts.dataset);
    auto labels = loadMnistLabels(opts.dataset);
    size_t correct = 0;
    std::vector<float> buf(MnistPixels);
    intptr_t inSizes[4] = {1, 1, 28, 28};
    static float outputData[10] __attribute__((aligned(64)));
    BorrowedBuffer<float, 2> output(outputData, sizesOutput);
    for (size_t i = 0; i < MnistCount; ++i) {
      fillMnistImage(buf.data(), images.data() + i * MnistPixels);
      dip::Image<float, 4> input(buf.data(), inSizes);
      _mlir_ciface_forward(&output, &paramsContainer, &weightsContainer,
                           &input);
      if (argmax(output.getData(), 10) == labels[i])
        ++correct;
    }
    std::cout << "top1=" << correct << "/10000" << std::endl;
    return 0;
  }

  static float inputData[MnistPixels] __attribute__((aligned(64)));
  intptr_t inputSizes[4] = {1, 1, 28, 28};
  loadLeNetInput(inputData);
  BorrowedImage input(inputData, inputSizes);
  static float outputData[10] __attribute__((aligned(64)));
  BorrowedBuffer<float, 2> output(outputData, sizesOutput);

  unsigned long start = read_cycles();
  _mlir_ciface_forward(&output, &paramsContainer, &weightsContainer, &input);
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

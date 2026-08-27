//===- buddy-mobilenetv3-main.cpp -----------------------------------------===//
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
#include <buddy/DIP/DIP.h>
#include <buddy/DIP/ImgContainer.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

constexpr size_t ParamsSize = 29136;
constexpr size_t WeightsSize = 2525832;
constexpr size_t ScalesSize = 224;
const std::string ImgName = "dog-32bit_224x224.bmp";

// Declare the mobilenet C interface.
extern "C" void _mlir_ciface_forward(MemRef<float, 2> *output,
                                     MemRef<float, 1> *arg0,
                                     MemRef<int8_t, 1> *weights,
                                     MemRef<float, 4> *input);

template <typename T, size_t N>
class BorrowedBuffer : public MemRef<T, N> {
public:
  BorrowedBuffer(T *data, intptr_t sizes[N]) : MemRef<T, N>(sizes, false, 0) {
    this->allocated = this->aligned = data;
  }
  ~BorrowedBuffer() { this->allocated = this->aligned = nullptr; }
};

template <typename T>
void loadBinary(const std::string &path, T *data, size_t count) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("failed to open binary file: " + path);
  file.read(reinterpret_cast<char *>(data), sizeof(T) * count);
  if (file.gcount() != static_cast<std::streamsize>(sizeof(T) * count))
    throw std::runtime_error("short binary file: " + path);
}

/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

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
  // std::string mobilenetDir = getenv("MOBILENETV3_DIR");
  std::string mobilenetDir = "./";
  std::ifstream in(mobilenetDir + "/Labels.txt");
  assert(in.is_open() && "Could not read the label file.");
  std::string label;
  for (int i = 0; i < idx; ++i)
    std::getline(in, label);
  std::getline(in, label);
  in.close();
  return label;
}

int main() {
  // Print the title of this example.
  const std::string title = "MobileNetV3 Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  // Define the sizes of the input and output tensors.
  intptr_t sizesOutput[2] = {1, 1000};

  // Create input and output containers for the image and model output.
  // std::string mobilenetDir = getenv("MOBILENETV3_DIR");
  std::string mobilenetDir = "./";
  std::string imgPath = mobilenetDir + "/images/" + ImgName;
  dip::Image<float, 4> input(imgPath, dip::DIP_RGB, true /* norm */);
  MemRef<float, 4> inputResize = dip::Resize4D_NCHW(
      &input, dip::INTERPOLATION_TYPE::BILINEAR_INTERPOLATION,
      {1, 3, 224, 224} /*{image_cols, image_rows}*/);

  MemRef<float, 2> output(sizesOutput);

  // Load model parameters from the specified file.
  static float paramsData[ParamsSize] __attribute__((aligned(64)));
  static int8_t weightsData[WeightsSize] __attribute__((aligned(64)));
  static uint8_t scalesData[ScalesSize] __attribute__((aligned(64)));
  intptr_t paramsSize[1] = {ParamsSize};
  intptr_t weightsSize[1] = {WeightsSize};
  BorrowedBuffer<float, 1> paramsContainer(paramsData, paramsSize);
  BorrowedBuffer<int8_t, 1> weightsContainer(weightsData, weightsSize);
  loadBinary(mobilenetDir + "/mobilenetv3.payload/params.f32", paramsData,
             ParamsSize);
  loadBinary(mobilenetDir + "/mobilenetv3.payload/weights.i8", weightsData,
             WeightsSize);
  loadBinary(mobilenetDir + "/mobilenetv3.payload/scales.bin", scalesData,
             ScalesSize);
  bb_mvin_mmio(reinterpret_cast<uintptr_t>(scalesData), 16,
               ScalesSize / 16, 16);
  
  unsigned long start = read_cycles();
  // Call the forward function of the model.
  _mlir_ciface_forward(&output, &paramsContainer, &weightsContainer,
                       &inputResize);
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

//===- buddy-alexnet-main.cpp ---------------------------------------------===//
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
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// Number of float32 parameters concatenated in arg0.data
// (201,838,952 = 5 conv + 3 fc weight/bias params, fp32).
constexpr size_t ParamsSize = 201838952;
const std::string ImgName = "dog-326x256.bmp";

// ImageNet mean/std used by the author's validation pipeline
// (see load_data.py in https://github.com/pie33000/alexnet).
constexpr float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
constexpr float IMAGENET_STD[3] = {0.229f, 0.224f, 0.225f};

// Fixed asset size: 326x256 (short side already resized to 256, so the
// Resize(256) -> CenterCrop(224) pipeline reduces to a pure center crop).
constexpr int64_t ImgHeight = 256;
constexpr int64_t ImgWidth = 326;
constexpr int64_t CropSize = 224;

// Declare the alexnet C interface.
extern "C" void _mlir_ciface_forward(MemRef<float, 2> *output,
                                     MemRef<float, 1> *arg0,
                                     MemRef<float, 4> *input);

/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

/// Load parameters into data container.
void loadParameters(const std::string &paramFilePath,
                    MemRef<float, 1> &params) {
  const auto loadStart = std::chrono::high_resolution_clock::now();
  // Open the parameter file in binary mode.
  std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
  if (!paramFile.is_open()) {
    throw std::runtime_error("[Error] Failed to open params file!");
  }
  printLogLabel();
  std::cout << "Loading params..." << std::endl;
  printLogLabel();
  // Print the canonical path of the parameter file.
  std::cout << "Params file: " << std::filesystem::canonical(paramFilePath)
            << std::endl;
  // Read the parameter data into the provided memory reference.
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
  std::string alexnetDir = "./";
  std::ifstream in(alexnetDir + "/Labels.txt");
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
  const std::string title = "AlexNet Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  // Define the sizes of the input and output tensors.
  intptr_t sizesOutput[2] = {1, 1000};

  // Create input and output containers for the image and model output.
  std::string alexnetDir = "./";
  std::string imgPath = alexnetDir + "/images/" + ImgName;
  dip::Image<float, 4> input(imgPath, dip::DIP_RGB, true /* norm */);
  if (input.getWidth() != ImgWidth || input.getHeight() != ImgHeight) {
    throw std::runtime_error(
        "[Error] Unexpected image size, expected 326x256: " + imgPath);
  }

  // Center-crop 224x224 and apply ImageNet normalization, reproducing the
  // author's validation preprocessing
  // (Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize).
  intptr_t sizesInput[4] = {1, 3, CropSize, CropSize};
  MemRef<float, 4> inputCrop(sizesInput);
  const int64_t yOffset = (ImgHeight - CropSize) / 2; // 16
  const int64_t xOffset = (ImgWidth - CropSize) / 2;  // 51
  const float *src = input.getData();
  float *dst = inputCrop.getData();
  const int64_t srcChannelStride = ImgHeight * ImgWidth;
  const int64_t dstChannelStride = CropSize * CropSize;
  for (int64_t c = 0; c < 3; ++c) {
    for (int64_t y = 0; y < CropSize; ++y) {
      for (int64_t x = 0; x < CropSize; ++x) {
        float pixel = src[c * srcChannelStride + (yOffset + y) * ImgWidth +
                          (xOffset + x)];
        dst[c * dstChannelStride + y * CropSize + x] =
            (pixel - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
      }
    }
  }

  MemRef<float, 2> output(sizesOutput);

  // Load model parameters from the specified file.
  std::string paramsDir = alexnetDir + "/arg0.data";
  MemRef<float, 1> paramsContainer({ParamsSize});
  loadParameters(paramsDir, paramsContainer);

  unsigned long start = read_cycles();
  _mlir_ciface_forward(&output, &paramsContainer, &inputCrop);
  unsigned long end = read_cycles();
  std::cout << "Cycle count: " << end - start << std::endl;

  auto out = output.getData();
  softmax(out, 1000);

  // Print the top-5 classification results.
  std::vector<std::pair<float, int>> sorted;
  sorted.reserve(1000);
  for (int i = 0; i < 1000; ++i)
    sorted.emplace_back(out[i], i);
  std::sort(sorted.begin(), sorted.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });
  for (int rank = 0; rank < 5; ++rank) {
    int idx = sorted[rank].second;
    std::cout << "Top " << rank + 1 << ": Index " << idx << ", Label \""
              << getLabel(idx) << "\", Probability " << sorted[rank].first
              << std::endl;
  }

  return 0;
}

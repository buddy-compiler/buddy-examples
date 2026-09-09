//===- buddy-yolo26n-main.cpp ---------------------------------------------===//
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

#include <algorithm>
#include <bbhw/isa/isa.h>
#include <buddy/Core/Container.h>
#include <buddy/DIP/DIP.h>
#include <buddy/DIP/ImgContainer.h>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <unistd.h>

#include "testutils.h"

constexpr size_t ParamsSize = 187536;
constexpr size_t WeightsSize = 21699632;
constexpr int InputSize = 640;
constexpr int MaxDetections = 300;
constexpr float PadValue = 114.0f / 255.0f;
constexpr float ScoreThreshold = 0.25f;

struct LetterboxResult {
  MemRef<float, 4> input;
  float scale;
  int padTop;
  int padLeft;
  int originalH;
  int originalW;
};

struct Detection {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  int classId;
};

extern "C" void _mlir_ciface_forward(MemRef<float, 3> *output,
                                     MemRef<float, 1> *arg0,
                                     MemRef<int8_t, 1> *weights,
                                     MemRef<float, 4> *arg1);

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
  const size_t bytes = sizeof(T) * count;
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0)
    throw std::runtime_error("failed to open binary file: " + path);
  if (read(fd, data, bytes) != static_cast<ssize_t>(bytes)) {
    close(fd);
    throw std::runtime_error("short binary file: " + path);
  }
  uint8_t extra;
  if (read(fd, &extra, 1) != 0) {
    close(fd);
    throw std::runtime_error("binary size mismatch: " + path);
  }
  close(fd);
}

void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

std::vector<std::string> loadLabels(const std::string &labelsFilePath) {
  std::ifstream labelsFile(labelsFilePath);
  if (!labelsFile.is_open()) {
    throw std::runtime_error("[Error] Failed to open labels file!");
  }

  std::vector<std::string> labels;
  std::string line;
  while (std::getline(labelsFile, line)) {
    if (!line.empty()) {
      labels.push_back(line);
    }
  }
  return labels;
}

LetterboxResult letterboxImage(dip::Image<float, 4> &image) {
  const int originalH = static_cast<int>(image.getSizes()[2]);
  const int originalW = static_cast<int>(image.getSizes()[3]);
  const float scale = std::min(static_cast<float>(InputSize) / originalH,
                               static_cast<float>(InputSize) / originalW);
  const int resizedH =
      std::max(1, static_cast<int>(std::round(originalH * scale)));
  const int resizedW =
      std::max(1, static_cast<int>(std::round(originalW * scale)));
  const int padTop = (InputSize - resizedH) / 2;
  const int padLeft = (InputSize - resizedW) / 2;

  MemRef<float, 4> resized = dip::Resize4D_NCHW(
      &image, dip::INTERPOLATION_TYPE::BILINEAR_INTERPOLATION,
      std::vector<uint>{1, 3, static_cast<uint>(resizedH),
                        static_cast<uint>(resizedW)});
  MemRef<float, 4> input({1, 3, InputSize, InputSize}, PadValue);

  float *dst = input.getData();
  float *src = resized.getData();
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < resizedH; ++y) {
      const size_t dstOffset =
          (static_cast<size_t>(c) * InputSize + (padTop + y)) * InputSize +
          padLeft;
      const size_t srcOffset =
          (static_cast<size_t>(c) * resizedH + y) * resizedW;
      std::memcpy(dst + dstOffset, src + srcOffset, sizeof(float) * resizedW);
    }
  }

  return {std::move(input), scale, padTop, padLeft, originalH, originalW};
}

std::vector<Detection> postprocess(MemRef<float, 3> &output) {
  const float *out = output.getData();
  const intptr_t *strides = output.getStrides();
  std::vector<Detection> candidates;
  const int detCount = static_cast<int>(output.getSizes()[1]);
  candidates.reserve(detCount);
  for (int i = 0; i < detCount; ++i) {
    const intptr_t base = i * strides[1];
    const float score = out[base + 4 * strides[2]];
    if (score >= ScoreThreshold) {
      candidates.push_back(
          {out[base + 0 * strides[2]], out[base + 1 * strides[2]],
           out[base + 2 * strides[2]], out[base + 3 * strides[2]], score,
           static_cast<int>(out[base + 5 * strides[2]])});
    }
  }
  return candidates;
}

Detection mapToOriginalImage(const Detection &det,
                             const LetterboxResult &letterbox) {
  const float invScale = 1.0f / letterbox.scale;
  Detection mapped = det;
  mapped.x1 = (det.x1 - letterbox.padLeft) * invScale;
  mapped.y1 = (det.y1 - letterbox.padTop) * invScale;
  mapped.x2 = (det.x2 - letterbox.padLeft) * invScale;
  mapped.y2 = (det.y2 - letterbox.padTop) * invScale;
  mapped.x1 =
      std::clamp(mapped.x1, 0.0f, static_cast<float>(letterbox.originalW - 1));
  mapped.y1 =
      std::clamp(mapped.y1, 0.0f, static_cast<float>(letterbox.originalH - 1));
  mapped.x2 =
      std::clamp(mapped.x2, 0.0f, static_cast<float>(letterbox.originalW - 1));
  mapped.y2 =
      std::clamp(mapped.y2, 0.0f, static_cast<float>(letterbox.originalH - 1));
  return mapped;
}

int main(int argc, char **argv) {
  const std::string title = "YOLO26n Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  const std::string imagePath =
      argc >= 2 ? argv[1] : "images/bus_16bit.bmp";
  const std::vector<std::string> labels = loadLabels("labels.txt");
  std::string imageExt = std::filesystem::path(imagePath).extension().string();
  std::transform(imageExt.begin(), imageExt.end(), imageExt.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (imageExt != ".bmp") {
    std::cerr << "Only .bmp image is supported in this example." << std::endl;
    return 1;
  }

  dip::Image<float, 4> image(imagePath, dip::DIP_RGB, true /* norm */);
  LetterboxResult letterbox = letterboxImage(image);

  static float paramsData[ParamsSize] __attribute__((aligned(64)));
  static int8_t weightsData[WeightsSize] __attribute__((aligned(64)));
  intptr_t paramsSize[1] = {ParamsSize};
  intptr_t weightsSize[1] = {WeightsSize};
  BorrowedBuffer<float, 1> paramsContainer(paramsData, paramsSize);
  BorrowedBuffer<int8_t, 1> weightsContainer(weightsData, weightsSize);
  loadBinary("yolo26.payload/params.f32", paramsData, ParamsSize);
  loadBinary("yolo26.payload/weights.i8", weightsData, WeightsSize);

  MemRef<float, 3> output({1, MaxDetections, 6});
  unsigned long start = read_cycles();
  _mlir_ciface_forward(&output, &paramsContainer, &weightsContainer,
                       &letterbox.input);
  unsigned long end = read_cycles();
  std::cout << "Cycle count: " << end - start << std::endl;

  const intptr_t *outSizes = output.getSizes();
  const intptr_t *outStrides = output.getStrides();
  printLogLabel();
  std::cout << "Output shape/stride: [" << outSizes[0] << ", " << outSizes[1]
            << ", " << outSizes[2] << "] / [" << outStrides[0] << ", "
            << outStrides[1] << ", " << outStrides[2] << "]" << std::endl;

  const std::vector<Detection> detections = postprocess(output);
  int validCount = 0;
  std::cout << std::fixed << std::setprecision(4);
  for (const Detection &det : detections) {
    if (det.score < ScoreThreshold) {
      continue;
    }
    const Detection mapped = mapToOriginalImage(det, letterbox);
    if (mapped.classId < 0 ||
        static_cast<size_t>(mapped.classId) >= labels.size()) {
      throw std::runtime_error("detection class_id out of labels range");
    }
    const std::string &label = labels[static_cast<size_t>(mapped.classId)];
    std::cout << "[" << validCount << "] class_id=" << mapped.classId
              << " label=" << label << " score=" << mapped.score << " box=("
              << mapped.x1 << ", " << mapped.y1 << ", " << mapped.x2 << ", "
              << mapped.y2 << ")" << std::endl;
    ++validCount;
  }
  std::cout << "Detections: " << validCount << std::endl;
  if (validCount == 0)
    throw std::runtime_error("FAIL expected detections > 0");
  // bus_16bit.bmp must contain a bus (COCO class 5); 8/29 golden was score~0.80.
  constexpr int kBus = 5;
  constexpr float kBusMin = 0.5f;
  bool gotBus = false;
  for (const Detection &det : detections) {
    if (det.score >= kBusMin && det.classId == kBus) {
      gotBus = true;
      break;
    }
  }
  if (!gotBus)
    throw std::runtime_error("FAIL expected bus class_id=5 score>=0.5");
  std::cout << "YOLO Inference PASS" << std::endl;
  return 0;
}

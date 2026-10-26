#include "SharedLibrary.h"
#include <algorithm>
#include <buddy/Core/Container.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
//===- buddy-yolo26n-runner.cpp
//---------------------------------------------===//
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
#include <buddy/Core/Container.h>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <unistd.h>
#include <vector>

#include "testutils.h"

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

using Forward = void (*)(MemRef<float, 3> *, MemRef<float, 1> *,
                         MemRef<float, 4> *);

template <typename T> MemRef<T, 1> loadBinary(const std::string &path) {
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
  std::cout << "\033[34;1m[Log] \033[0mLoad time: " << loadTime.count() << "s"
            << std::endl;
  return tensor;
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

LetterboxResult letterboxImage(const cv::Mat &image) {
  const int originalH = image.rows;
  const int originalW = image.cols;
  const double scale = std::min(static_cast<double>(InputSize) / originalH,
                                static_cast<double>(InputSize) / originalW);
  const int resizedH = static_cast<int>(std::nearbyint(originalH * scale));
  const int resizedW = static_cast<int>(std::nearbyint(originalW * scale));
  const int padTop = (InputSize - resizedH) / 2;
  const int padLeft = (InputSize - resizedW) / 2;

  // Resize bytes before normalization, matching the PyTorch image pipeline.
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(resizedW, resizedH), 0, 0,
             cv::INTER_LINEAR);
  MemRef<float, 4> input({1, 3, InputSize, InputSize}, PadValue);
  float *dst = input.getData();
  for (int y = 0; y < resizedH; ++y) {
    const auto *row = resized.ptr<cv::Vec3b>(y);
    for (int x = 0; x < resizedW; ++x) {
      for (int c = 0; c < 3; ++c) {
        const size_t offset =
            (static_cast<size_t>(c) * InputSize + padTop + y) * InputSize +
            padLeft + x;
        dst[offset] = row[x][2 - c] / 255.0f;
      }
    }
  }
  return {std::move(input), static_cast<float>(scale),
          padTop,           padLeft,
          originalH,        originalW};
}

std::vector<Detection> postprocess(MemRef<float, 3> &output) {
  const float *out = output.getData();
  const intptr_t *strides = output.getStrides();
  std::vector<Detection> candidates;
  const int detCount = static_cast<int>(output.getSizes()[1]);
  candidates.reserve(detCount);
  for (int i = 0; i < detCount; ++i) {
    const intptr_t base = i * strides[1];
    for (int classId = 0; classId < output.getSizes()[2] - 4; ++classId) {
      const float score = out[base + (4 + classId) * strides[2]];
      if (score >= ScoreThreshold)
        candidates.push_back({out[base], out[base + strides[2]],
                              out[base + 2 * strides[2]],
                              out[base + 3 * strides[2]], score, classId});
    }
  }
  const size_t count = std::min(candidates.size(), size_t(MaxDetections));
  std::partial_sort(
      candidates.begin(), candidates.begin() + count, candidates.end(),
      [](const Detection &a, const Detection &b) { return a.score > b.score; });
  candidates.resize(count);
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

namespace {
class YoloRunner final : public buddy::runtime::InferenceRunner {
public:
  void run(const buddy::runtime::RunConfig &config) override {
    const auto manifest =
        buddy::runtime::ModelManifest::loadFromRax(config.raxPath);
    std::vector<model::SharedLibrary> runtimes;
    for (const auto &path : manifest.dependentSoPaths)
      runtimes.push_back(model::openLibrary(path));
    auto library = model::openLibrary(manifest.soPath);
    auto forward =
        reinterpret_cast<Forward>(dlsym(library.get(), "_mlir_ciface_forward"));
    if (!forward)
      throw std::runtime_error("Missing _mlir_ciface_forward: " +
                               std::string(dlerror()));
    const std::string imagePath = config.imagePath;
    const auto labels = loadLabels(manifest.weightPaths.at(1));
    const auto image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty())
      throw std::runtime_error("Cannot load image: " + imagePath);
    LetterboxResult letterbox = letterboxImage(image);

    auto paramsContainer = loadBinary<float>(manifest.weightPaths.at(0));

    MemRef<float, 3> output(std::vector<size_t>{1, 8400, 84}, false, 0);
    std::cout << "\033[34;1m[Log] \033[0mStarting inference..." << std::endl;
    unsigned long start = read_counter();
    forward(&output, &paramsContainer, &letterbox.input);
    unsigned long end = read_counter();
    std::cout << "Counter: " << end - start << std::endl;

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
  }
};

} // namespace
extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new YoloRunner();
}
extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

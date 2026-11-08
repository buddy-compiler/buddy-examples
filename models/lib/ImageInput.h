//===- ImageInput.h -------------------------------------------------------===//
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

#ifndef BUDDY_MODELS_IMAGE_INPUT_H
#define BUDDY_MODELS_IMAGE_INPUT_H

#include <buddy/Core/Container.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>

namespace model {
inline MemRef<float, 4> loadImageNetImage(const std::string &path) {
  const auto image = cv::imread(path, cv::IMREAD_COLOR);
  if (image.empty())
    throw std::runtime_error("Cannot load image: " + path);
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
  MemRef<float, 4> input({1, 3, 224, 224});
  constexpr float mean[] = {0.485f, 0.456f, 0.406f};
  constexpr float deviation[] = {0.229f, 0.224f, 0.225f};
  for (int y = 0; y < 224; ++y) {
    const auto *row = resized.ptr<cv::Vec3b>(y);
    for (int x = 0; x < 224; ++x)
      for (int c = 0; c < 3; ++c)
        input.getData()[(c * 224 + y) * 224 + x] =
            (row[x][2 - c] / 255.0f - mean[c]) / deviation[c];
  }
  return input;
}
} // namespace model
#endif

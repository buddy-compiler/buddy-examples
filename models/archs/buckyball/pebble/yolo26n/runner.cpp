#include "SharedLibrary.h"
#include "model.h"
#include "testutils.h"
#include <buddy/Core/Container.h>
#include <buddy/DIP/DIP.h>
#include <buddy/DIP/ImgContainer.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using model::openLibrary;
using model::SharedLibrary;

using yolo::Forward;

constexpr size_t InputSize = 640;
constexpr size_t Detections = 300;
constexpr float ScoreThreshold = 0.25f;

template <typename T> MemRef<T, 1> loadParameters(const std::string &path) {
  std::ifstream file;
  file.exceptions(std::ios::failbit | std::ios::badbit);
  file.open(path, std::ios::binary | std::ios::ate);
  const size_t bytes = file.tellg();
  if (bytes % sizeof(T))
    throw std::runtime_error("Invalid parameter byte count: " + path);
  MemRef<T, 1> parameters(std::vector<size_t>{bytes / sizeof(T)});
  file.seekg(0);
  file.read(reinterpret_cast<char *>(parameters.getData()), bytes);
  return parameters;
}

std::vector<std::string> loadLabels(const std::string &path) {
  std::ifstream file;
  file.exceptions(std::ios::badbit);
  file.open(path);
  std::vector<std::string> labels;
  for (std::string line; std::getline(file, line);)
    labels.push_back(line);
  return labels;
}

class YoloRunner final : public buddy::runtime::InferenceRunner {
public:
  void run(const buddy::runtime::RunConfig &config) override {
    const auto manifest =
        buddy::runtime::ModelManifest::loadFromRax(config.raxPath);
    std::vector<SharedLibrary> runtimes;
    for (const auto &path : manifest.dependentSoPaths)
      runtimes.push_back(openLibrary(path));
    auto library = openLibrary(manifest.soPath);
    const auto &entry = manifest.codeObjects.at(0).entrySymbol;
    auto forward =
        reinterpret_cast<Forward>(dlsym(library.get(), entry.c_str()));
    if (!forward)
      throw std::runtime_error("Missing " + entry + ": " +
                               std::string(dlerror()));
    auto parameters = loadParameters<float>(manifest.weightPaths.at(0));
    auto quantizedBias = loadParameters<int32_t>(manifest.weightPaths.at(1));
    auto quantizedWeights = loadParameters<int8_t>(manifest.weightPaths.at(2));
    const auto labels = loadLabels(manifest.vocabPath);

    // The prepared image of the recipe against its reference, as
    // baremetal.cpp checks it: the same lines, and a failure when the boxes
    // differ.
    if (!config.referencePath.empty()) {
      auto prepared = loadParameters<float>(config.inputPath);
      auto reference = loadParameters<float>(config.referencePath);
      if (prepared.getSize() != 640 * 640 * 3 ||
          reference.getSize() != Detections * 6)
        throw std::runtime_error("Invalid prepared image or reference size");
      yolo::Parameters f32{parameters.getData(),
                           parameters.getData(),
                           0,
                           {int64_t(parameters.getSize())},
                           {1}};
      yolo::Bias i32{quantizedBias.getData(),
                     quantizedBias.getData(),
                     0,
                     {int64_t(quantizedBias.getSize())},
                     {1}};
      yolo::Weights i8{quantizedWeights.getData(),
                       quantizedWeights.getData(),
                       0,
                       {int64_t(quantizedWeights.getSize())},
                       {1}};
      yolo::Input imageInput{prepared.getData(),
                             prepared.getData(),
                             0,
                             {1, 640, 640, 3},
                             {1228800, 1920, 3, 1}};
      yolo::Output output{};
      const uint64_t start = read_counter();
      forward(&output, &f32, &i32, &i8, &imageInput);
      const uint64_t elapsed = read_counter() - start;
      const float *expected = reference.getData();
      unsigned count = 0, gold = 0;
      for (unsigned i = 0; i < Detections; ++i) {
        count += output.data()[i * 6 + 4] >= 0.25f;
        gold += expected[i * 6 + 4] >= 0.25f;
      }
      printf("Detections: %u, reference %u\nInference cycles: %lu\n", count,
             gold, static_cast<unsigned long>(elapsed));
      const int failed =
          check_output(output.data(), expected, Detections * 6, 1e-5f) ||
          count != gold;
      std::free(output.allocated);
      if (failed)
        throw std::runtime_error("Output differs from the reference");
      return;
    }

    // Letterbox: scale the image into 640x640 and pad it with gray.
    dip::Image<float, 4> image(config.imagePath, dip::DIP_RGB, true /* norm */);
    const size_t height = image.getSizes()[2], width = image.getSizes()[3];
    const float scale =
        std::min(float(InputSize) / height, float(InputSize) / width);
    const size_t rows = std::lround(height * scale);
    const size_t columns = std::lround(width * scale);
    const size_t top = (InputSize - rows) / 2, left = (InputSize - columns) / 2;
    MemRef<float, 4> resized = dip::Resize4D_NCHW(
        &image, dip::INTERPOLATION_TYPE::BILINEAR_INTERPOLATION,
        {1, 3, uint(rows), uint(columns)});
    MemRef<float, 4> input({1, InputSize, InputSize, 3}, 114.0f / 255.0f);
    for (size_t y = 0; y < rows; ++y)
      for (size_t x = 0; x < columns; ++x)
        for (size_t c = 0; c < 3; ++c)
          input.getData()[((top + y) * InputSize + left + x) * 3 + c] =
              resized.getData()[(c * rows + y) * columns + x];

    // The compiled function returns an owned memref descriptor of the 300
    // best boxes, [x1, y1, x2, y2, score, class] in input pixels.
    yolo::Parameters f32{parameters.getData(),
                         parameters.getData(),
                         0,
                         {int64_t(parameters.getSize())},
                         {1}};
    yolo::Bias i32{quantizedBias.getData(),
                   quantizedBias.getData(),
                   0,
                   {int64_t(quantizedBias.getSize())},
                   {1}};
    yolo::Weights i8{quantizedWeights.getData(),
                     quantizedWeights.getData(),
                     0,
                     {int64_t(quantizedWeights.getSize())},
                     {1}};
    yolo::Input imageInput{input.getData(),
                           input.getData(),
                           0,
                           {1, 640, 640, 3},
                           {1228800, 1920, 3, 1}};
    yolo::Output output{};
    const auto start = read_time_ns();
    forward(&output, &f32, &i32, &i8, &imageInput);
    const double elapsed = (read_time_ns() - start) * 1e-9;
    const float *boxes = output.data();
    size_t count = 0;
    for (size_t i = 0; i < Detections; ++i) {
      const float *box = boxes + i * 6;
      if (box[4] < ScoreThreshold)
        continue;
      auto x = [&](float v) {
        return std::clamp((v - left) / scale, 0.0f, float(width - 1));
      };
      auto y = [&](float v) {
        return std::clamp((v - top) / scale, 0.0f, float(height - 1));
      };
      std::cout << "[" << count++ << "] label=" << labels.at(size_t(box[5]))
                << " score=" << box[4] << " box=(" << x(box[0]) << ", "
                << y(box[1]) << ", " << x(box[2]) << ", " << y(box[3]) << ")\n";
    }
    std::cout << "Detections: " << count << '\n';
    if (!config.outputPath.empty()) {
      std::ofstream dump;
      dump.exceptions(std::ios::failbit | std::ios::badbit);
      dump.open(config.outputPath, std::ios::binary);
      dump.write(reinterpret_cast<const char *>(boxes),
                 Detections * 6 * sizeof(float));
      dump.close();
    }
    std::free(output.allocated);
    if (!config.suppressStats)
      std::cout << "Inference time: " << elapsed << "s\n";
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

#include "testutils.h"
#include <buddy/Core/Container.h>
#include <buddy/DIP/ImgContainer.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void _mlir_ciface_forward(MemRef<float, 2> *, MemRef<float, 1> *,
                                     MemRef<int8_t, 1> *, MemRef<float, 4> *);

template <typename T> MemRef<T, 1> load(const std::string &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  stream.exceptions(std::ios::failbit | std::ios::badbit);
  const size_t bytes = stream.tellg();
  if (bytes % sizeof(T)) throw std::runtime_error("invalid tensor size: " + path);
  MemRef<T, 1> result(std::vector<size_t>{bytes / sizeof(T)});
  stream.seekg(0);
  stream.read(reinterpret_cast<char *>(result.getData()), bytes);
  return result;
}

int main(int argc, char **argv) {
  if (argc != 2 && !(argc == 3 && std::string(argv[1]) == "--tensor"))
    throw std::runtime_error("expected a 224x224 image, or --tensor input.f32");
  MemRef<float, 4> input(std::vector<size_t>{1, 3, 224, 224});
  if (argc == 3) {
    auto values = load<float>(argv[2]);
    if (values.getSize() != input.getSize())
      throw std::runtime_error("expected 1x3x224x224 input tensor");
    std::copy_n(values.getData(), input.getSize(), input.getData());
  } else {
    dip::Image<float, 4> image(argv[1], dip::DIP_RGB, true);
    if (image.getSizes()[2] != 224 || image.getSizes()[3] != 224)
      throw std::runtime_error("expected a 224x224 RGB image");
    constexpr float mean[] = {0.485f, 0.456f, 0.406f};
    constexpr float scale[] = {0.229f, 0.224f, 0.225f};
    for (size_t channel = 0; channel < 3; ++channel)
      for (size_t pixel = 0; pixel < 224 * 224; ++pixel) {
        size_t index = channel * 224 * 224 + pixel;
        input.getData()[index] = (image.getData()[index] - mean[channel]) / scale[channel];
      }
  }
  auto params = load<float>("alexnet.payload/params.f32");
  auto weights = load<int8_t>("alexnet.payload/weights.bin");
  MemRef<float, 2> output(std::vector<size_t>{1, 1000});
  unsigned long start = read_cycles();
  _mlir_ciface_forward(&output, &params, &weights, &input);
  std::cout << "Cycle count: " << read_cycles() - start << '\n';
  for (size_t index = 0; index < 1000; ++index)
    if (!std::isfinite(output.getData()[index]))
      throw std::runtime_error("AlexNet produced non-finite logits");
  std::ofstream logits("logits.f32", std::ios::binary);
  logits.exceptions(std::ios::failbit | std::ios::badbit);
  logits.write(reinterpret_cast<const char *>(output.getData()), 1000 * sizeof(float));
  std::vector<size_t> order(1000);
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                    [&](size_t a, size_t b) { return output.getData()[a] > output.getData()[b]; });
  std::cout << "Classification Index: " << order[0] << "\nTop5:";
  for (size_t index = 0; index < 5; ++index) std::cout << ' ' << order[index];
  std::cout << '\n';
}

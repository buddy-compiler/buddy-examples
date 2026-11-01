#include "model.h"
#include "testutils.h"

#include <stdio.h>
#include <stdlib.h>

alignas(64) static const unsigned char parameters[] = {
#embed "arg0.data"
};
alignas(64) static const unsigned char biases[] = {
#embed "arg1.data"
};
alignas(64) static const unsigned char weights[] = {
#embed "arg2.data"
};
alignas(64) static const unsigned char pixels[] = {
#embed "inputs.data"
};
static_assert(sizeof(pixels) == 3 * 640 * 640 * sizeof(float));

int main() {
  const auto *p = reinterpret_cast<const float *>(parameters);
  const auto *b = reinterpret_cast<const int32_t *>(biases);
  const auto *w = reinterpret_cast<const int8_t *>(weights);
  const auto *image = reinterpret_cast<const float *>(pixels);
  yolo::Parameters f32{p, p, 0, {sizeof(parameters) / sizeof(float)}, {1}};
  yolo::Bias i32{b, b, 0, {sizeof(biases) / sizeof(int32_t)}, {1}};
  yolo::Weights i8{w, w, 0, {sizeof(weights)}, {1}};
  yolo::Input input{
      image, image, 0, {1, 3, 640, 640}, {1228800, 409600, 640, 1}};
  yolo::Output output{};
  const uint64_t start = read_counter();
  _mlir_ciface_forward(&output, &f32, &i32, &i8, &input);
  const uint64_t elapsed = read_counter() - start;
  unsigned count = 0;
  for (unsigned i = 0; i < 300; ++i) {
    const float *box = output.data() + i * 6;
    if (box[4] >= 0.25f)
      ++count;
  }
  printf("Detections: %u\nInference cycles: %lu\n", count,
         static_cast<unsigned long>(elapsed));
  free(output.allocated);
  return 0;
}

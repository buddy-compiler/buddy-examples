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
// The 300 detections [x1, y1, x2, y2, score, class] of the integer
// reference model on that image.
alignas(64) static const unsigned char reference[] = {
#embed "reference.data"
};
static_assert(sizeof(reference) == 300 * 6 * sizeof(float));

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
  const auto *expected = reinterpret_cast<const float *>(reference);
  unsigned count = 0, gold = 0;
  for (unsigned i = 0; i < 300; ++i) {
    count += output.data()[i * 6 + 4] >= 0.25f;
    gold += expected[i * 6 + 4] >= 0.25f;
  }
  printf("Detections: %u, reference %u\nInference cycles: %lu\n", count, gold,
         static_cast<unsigned long>(elapsed));
  // The device computes the reference's integers; only FP32 rounding of the
  // boxes and scores may differ.
  const int failed =
      check_output(output.data(), expected, 300 * 6, 1e-5f) || count != gold;
  free(output.allocated);
  return failed;
}

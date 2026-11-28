#include "ModelImage.h"
#include "model.h"
#include "testutils.h"
#include <string.h>

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
int main() {
  const auto *p = reinterpret_cast<const float *>(parameters);
  const auto *b = reinterpret_cast<const int32_t *>(biases);
  const auto *w = reinterpret_cast<const int8_t *>(weights);
  uint64_t bytes;
  const auto *ids = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_IMAGE, "labels.data", &bytes));
  const size_t samples = bytes / sizeof(int64_t);
  const auto *images = reinterpret_cast<const float *>(
      model::imageFile(MODEL_IMAGE, "pixels.data", &bytes));
  const auto *reference = reinterpret_cast<const float *>(
      model::imageFile(MODEL_IMAGE, "reference.data", &bytes));
  yolo::Parameters f32{p, p, 0, {sizeof(parameters) / sizeof(float)}, {1}};
  yolo::Bias i32{b, b, 0, {sizeof(biases) / sizeof(int32_t)}, {1}};
  yolo::Weights i8{w, w, 0, {sizeof(weights)}, {1}};
  size_t passed = 0;
  for (size_t sample = 0; sample < samples; ++sample) {
    const float *image = images + sample * 3 * 640 * 640;
    yolo::Input input{image, image, 0, {1, 640, 640, 3}, {1228800, 1920, 3, 1}};
    yolo::Output output{};
    _mlir_ciface_forward(&output, &f32, &i32, &i8, &input);
    const float *expected = reference + sample * 300 * 6;
    unsigned count = 0, gold = 0;
    for (unsigned i = 0; i < 300; ++i) {
      count += output.data()[i * 6 + 4] >= 0.25f;
      gold += expected[i * 6 + 4] >= 0.25f;
    }
    passed +=
        !check_output(output.data(), expected, 300 * 6, 1e-5f) && count == gold;
    // Encode FP32 words exactly so the host can score all images together.
    printf("YOLO boxes %ld ", (long)ids[sample]);
    for (unsigned i = 0; i < 300 * 6; ++i) {
      uint32_t bits;
      memcpy(&bits, &output.data()[i], sizeof(bits));
      printf("%08x", (unsigned)bits);
    }
    printf("\n");
    free(output.allocated);
  }
  printf("Detection dataset: %lu samples, %lu passed\n", (unsigned long)samples,
         (unsigned long)passed);
  return 0;
}

#include "ImageClassifier.h"
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
static_assert(sizeof(pixels) == 224 * 224 * 3 * sizeof(float));

int main() {
  const auto *p = reinterpret_cast<const float *>(parameters);
  const auto *b = reinterpret_cast<const int32_t *>(biases);
  const auto *w = reinterpret_cast<const int8_t *>(weights);
  const auto *image = reinterpret_cast<const float *>(pixels);
  model::Parameters f32{p, p, 0, {sizeof(parameters) / sizeof(float)}, {1}};
  model::Bias i32{b, b, 0, {sizeof(biases) / sizeof(int32_t)}, {1}};
  model::Weights i8{w, w, 0, {sizeof(weights)}, {1}};
  model::Image input{image, image, 0, {1, 224, 224, 3}, {150528, 672, 3, 1}};
  model::Scores output{};
  const uint64_t start = read_counter();
  _mlir_ciface_forward(&output, &f32, &i32, &i8, &input);
  const uint64_t elapsed = read_counter() - start;
  const float *values = output.data();
  unsigned best = 0;
  for (unsigned i = 1; i < 1000; ++i)
    if (values[i] > values[best])
      best = i;
  printf("Classification Index: %u\nInference cycles: %lu\n", best,
         static_cast<unsigned long>(elapsed));
  free(output.allocated);
  return 0;
}

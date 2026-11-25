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
// One prepared NHWC image supplied with the baremetal payload.
alignas(64) static const unsigned char pixels[] = {
#embed "inputs.data"
};
static_assert(sizeof(pixels) == 28 * 28 * sizeof(float));
// The logits of the fixed-point reference model on that image.
alignas(64) static const unsigned char reference[] = {
#embed "reference.data"
};
static_assert(sizeof(reference) == 10 * sizeof(float));

int main() {
  const auto *p = reinterpret_cast<const float *>(parameters);
  const auto *b = reinterpret_cast<const int32_t *>(biases);
  const auto *w = reinterpret_cast<const int8_t *>(weights);
  const auto *image = reinterpret_cast<const float *>(pixels);
  lenet::Parameters f32{p, p, 0, {sizeof(parameters) / 4}, {1}};
  lenet::Bias i32{b, b, 0, {sizeof(biases) / 4}, {1}};
  lenet::Weights i8{w, w, 0, {sizeof(weights)}, {1}};
  lenet::Input input{image, image, 0, {1, 28, 28, 1}, {784, 28, 1, 1}};
  lenet::Output output{};
  const uint64_t start = read_counter();
  _mlir_ciface_forward(&output, &f32, &i32, &i8, &input);
  const uint64_t elapsed = read_counter() - start;
  const float *values = output.data();
  const auto *expected = reinterpret_cast<const float *>(reference);
  unsigned best = 0, gold = 0;
  for (unsigned i = 1; i < 10; ++i) {
    if (values[i] > values[best])
      best = i;
    if (expected[i] > expected[gold])
      gold = i;
  }
  printf("Classification: %u, reference %u\nInference cycles: %lu\n", best,
         gold, static_cast<unsigned long>(elapsed));
  // The device computes the reference's integers; only the FP32 dequant of
  // the logits may round differently.
  const int failed = check_output(values, expected, 10, 1e-5f) || best != gold;
  free(output.allocated);
  return failed;
}

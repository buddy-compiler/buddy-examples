// MobileNetV3-Small on the toy hart, bare metal: one prepared image against
// the logits of the calibrated INT8 model in PyTorch.
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
// One prepared NCHW image supplied with the baremetal payload.
alignas(64) static const unsigned char pixels[] = {
#embed "inputs.data"
};
static_assert(sizeof(pixels) == 3 * 224 * 224 * sizeof(float));
alignas(64) static const unsigned char reference[] = {
#embed "reference.data"
};
static_assert(sizeof(reference) == 1000 * sizeof(float));

int main() {
  const auto *p = reinterpret_cast<const float *>(parameters);
  const auto *b = reinterpret_cast<const int32_t *>(biases);
  const auto *w = reinterpret_cast<const int8_t *>(weights);
  mobilenet::Bias i32{b, b, 0, {sizeof(biases) / sizeof(int32_t)}, {1}};
  mobilenet::Weights i8{w, w, 0, {sizeof(weights)}, {1}};
  const auto *image = reinterpret_cast<const float *>(pixels);
  mobilenet::Parameters f32{p, p, 0, {sizeof(parameters) / 4}, {1}};
  mobilenet::Input input{
      image, image, 0, {1, 3, 224, 224}, {150528, 50176, 224, 1}};
  mobilenet::Output output{};
  const uint64_t start = read_counter();
  _mlir_ciface_forward(&output, &f32, &i32, &i8, &input);
  const uint64_t elapsed = read_counter() - start;
  const float *values = output.data();
  const auto *expected = reinterpret_cast<const float *>(reference);
  unsigned best = 0, gold = 0;
  for (unsigned i = 1; i < 1000; ++i) {
    if (values[i] > values[best])
      best = i;
    if (expected[i] > expected[gold])
      gold = i;
  }
  printf("Classification: %u, reference %u\nInference cycles: %lu\n", best,
         gold, static_cast<unsigned long>(elapsed));
  // An INT8 code on a rounding boundary flips with the order of the FP32 sums
  // before it (the software depthwise filters and gates against PyTorch): FP32
  // noise of 2e-7 there moves the reference's logits by up to 0.040 of the
  // largest one, the class unchanged. The bound is 0.05 and the class must
  // match.
  const int failed =
      check_output(values, expected, 1000, 5e-2f) || best != gold;
  free(output.allocated);
  return failed;
}

// MobileNetV3-Small compiled for Gemmini, bare metal on the Gemmini device
// (Rocket and Gemmini RTL): the first check image against the quantized model
// in software (buddy-pytorch-calibrate.py), which the compiled model on spike
// with the Gemmini extension matches bit for bit.
#include "testutils.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// A memref of the compiled graph.
template <typename T, int N> struct Buffer {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};

// The parameters are grouped by type: FP32, INT32 biases, INT8 weights.
extern "C" void _mlir_ciface_forward(Buffer<float, 2> *output,
                                     Buffer<float, 1> *parameters,
                                     Buffer<int32_t, 1> *bias,
                                     Buffer<int8_t, 1> *weights,
                                     Buffer<float, 4> *input);

alignas(64) static const unsigned char parameter_data[] = {
#embed "arg0.data"
};
alignas(64) static const unsigned char bias_data[] = {
#embed "arg1.data"
};
alignas(64) static const unsigned char weight_data[] = {
#embed "arg2.data"
};
// The first check image and its reference logits.
alignas(64) static const unsigned char image_data[] = {
#embed "check-images.data" limit(3 * 224 * 224 * 4)
};
alignas(64) static const unsigned char logit_data[] = {
#embed "check-logits.data" limit(1000 * 4)
};
// The logits of every check image, compared at once at the end.
static float logits[1 * 1000];

int main() {
  auto *values = (float *)parameter_data;
  auto *biases = (int32_t *)bias_data;
  auto *weights = (int8_t *)weight_data;
  const auto *expected = reinterpret_cast<const float *>(logit_data);
  Buffer<float, 1> parameters{
      values, values, 0, {intptr_t(sizeof(parameter_data) / 4)}, {1}};
  Buffer<int32_t, 1> bias{
      biases, biases, 0, {intptr_t(sizeof(bias_data) / 4)}, {1}};
  Buffer<int8_t, 1> quantized{
      weights, weights, 0, {intptr_t(sizeof(weight_data))}, {1}};
  const unsigned images = sizeof(image_data) / (3 * 224 * 224 * 4);
  unsigned same = 0;
  uint64_t total = 0;
  for (unsigned image = 0; image < images; ++image) {
    auto *pixels = (float *)image_data + image * 3 * 224 * 224;
    Buffer<float, 4> input{
        pixels, pixels, 0, {1, 3, 224, 224}, {150528, 50176, 224, 1}};
    // Each invocation returns an owned output descriptor.
    Buffer<float, 2> output;
    const uint64_t start = read_counter();
    _mlir_ciface_forward(&output, &parameters, &bias, &quantized, &input);
    total += read_counter() - start;
    const float *got = output.aligned + output.offset;
    const float *want = expected + image * 1000;
    unsigned best = 0, reference = 0;
    for (unsigned i = 0; i < 1000; ++i) {
      best = got[i] > got[best] ? i : best;
      reference = want[i] > want[reference] ? i : reference;
      logits[image * 1000 + i] = got[i];
    }
    same += best == reference;
    printf("image %u: class %u (reference %u)\n", image, best, reference);
    free(output.allocated);
  }
  printf("Inference cycles: %lu\n", (unsigned long)(total / images));
  // The device computes the reference's INT8 products and INT32 sums; the
  // logits match it to the last bits of their FP32 dequant.
  const int failed = check_output(logits, expected, images * 1000, 1e-5f);
  printf("Top-1: %u/%u as the quantized model\n", same, images);
  return failed || same != images;
}

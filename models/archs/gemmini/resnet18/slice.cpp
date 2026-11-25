// A slice of ResNet18 small enough for an RTL simulation of the Gemmini
// device: the tail of the model (layer4[1], avgpool, fc) compiled as the model
// is, bare metal on Rocket with Gemmini. Its input is the reference's layer4[0]
// output for the first check image, its result the model's logits for that
// image, checked bit for bit (buddy-pytorch-slice.py).
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
alignas(64) static const unsigned char input_data[] = {
#embed "check-input.data"
};
alignas(64) static const unsigned char output_data[] = {
#embed "check-output.data"
};
constexpr unsigned channels = 512, size = 7, logits = 1000;
static_assert(sizeof(input_data) == channels * size * size * 4);
static_assert(sizeof(output_data) == logits * 4);

static uint64_t cycles() {
  uint64_t value;
  asm volatile("rdcycle %0" : "=r"(value));
  return value;
}

int main() {
  auto *floats = (float *)parameter_data;
  auto *biases = (int32_t *)bias_data;
  auto *weights = (int8_t *)weight_data;
  auto *values = (float *)input_data;
  const auto *expected = reinterpret_cast<const float *>(output_data);
  Buffer<float, 1> parameters{
      floats, floats, 0, {intptr_t(sizeof(parameter_data) / 4)}, {1}};
  Buffer<int32_t, 1> bias{
      biases, biases, 0, {intptr_t(sizeof(bias_data) / 4)}, {1}};
  Buffer<int8_t, 1> quantized{
      weights, weights, 0, {intptr_t(sizeof(weight_data))}, {1}};
  Buffer<float, 4> input{values,
                         values,
                         0,
                         {1, channels, size, size},
                         {channels * size * size, size * size, size, 1}};
  // The invocation returns an owned output descriptor.
  Buffer<float, 2> output;
  const uint64_t start = cycles();
  _mlir_ciface_forward(&output, &parameters, &bias, &quantized, &input);
  const uint64_t elapsed = cycles() - start;
  const float *got = output.aligned + output.offset;
  unsigned wrong = 0, best = 0, reference = 0;
  for (unsigned i = 0; i < logits; ++i) {
    best = got[i] > got[best] ? i : best;
    reference = expected[i] > expected[reference] ? i : reference;
    if (got[i] != expected[i]) {
      if (!wrong)
        printf("resnet18 slice: logit %u is %f, not %f\n", i, (double)got[i],
               (double)expected[i]);
      ++wrong;
    }
  }
  printf(
      "resnet18 slice (layer4[1], avgpool, fc) on the gemmini device: class %u "
      "(model %u), "
      "%u of %u logits as the model's, %lu cycles\n",
      best, reference, logits - wrong, logits, (unsigned long)elapsed);
  free(output.allocated);
  return wrong != 0;
}

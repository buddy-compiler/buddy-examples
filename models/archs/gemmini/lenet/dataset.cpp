// LeNet compiled for Gemmini, the accuracy run: the MNIST test images of the
// model region (a shard of the test set; bench-device --bemu-accuracy packs it
// into the DDR image of each run). Prints one line: the samples, those
// classified as labelled (task accuracy), those predicting what the quantized
// model in software predicts (the pass rate) and those that model classifies
// as labelled (the accuracy of PyTorch on the same samples).
#include "ModelImage.h"
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
// The check images and the reference logits of the first four.
int main() {
  auto *values = (float *)parameter_data;
  auto *biases = (int32_t *)bias_data;
  auto *weights = (int8_t *)weight_data;
  Buffer<float, 1> parameters{
      values, values, 0, {intptr_t(sizeof(parameter_data) / 4)}, {1}};
  Buffer<int32_t, 1> bias{
      biases, biases, 0, {intptr_t(sizeof(bias_data) / 4)}, {1}};
  Buffer<int8_t, 1> quantized{
      weights, weights, 0, {intptr_t(sizeof(weight_data))}, {1}};
  // The samples of this run, in the model region.
  uint64_t bytes;
  const auto *images = reinterpret_cast<const float *>(
      model::imageFile(MODEL_REGION, "inputs.data", &bytes));
  const auto *references = reinterpret_cast<const float *>(
      model::imageFile(MODEL_REGION, "reference.data", &bytes));
  const auto *labels = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_REGION, "labels.data", &bytes));
  const uint64_t samples = bytes / sizeof(int64_t);
  uint64_t correct = 0, passed = 0, golden = 0;
  for (uint64_t sample = 0; sample < samples; ++sample) {
    auto *pixels = const_cast<float *>(images) + sample * 28 * 28;
    Buffer<float, 4> input{
        pixels, pixels, 0, {1, 1, 28, 28}, {784, 784, 28, 1}};
    Buffer<float, 2> output;
    _mlir_ciface_forward(&output, &parameters, &bias, &quantized, &input);
    const float *got = output.aligned + output.offset;
    const float *want = references + sample * 10;
    unsigned best = 0, gold = 0;
    for (unsigned i = 0; i < 10; ++i) {
      best = got[i] > got[best] ? i : best;
      gold = want[i] > want[gold] ? i : gold;
    }
    correct += best == labels[sample];
    golden += gold == labels[sample];
    // A sample passes when it predicts what the reference predicts.
    passed += best == gold;
    free(output.allocated);
  }
  printf("Dataset: %lu samples, %lu correct, %lu passed, %lu reference "
         "correct\n",
         (unsigned long)samples, (unsigned long)correct, (unsigned long)passed,
         (unsigned long)golden);
  return 0;
}

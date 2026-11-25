// AlexNet compiled for Gemmini, the accuracy run: the ImageNet validation
// images of the model region (a shard of the set; bench-device
// --bemu-accuracy packs it into the DDR image of each run), each normalized
// as the check images are. Prints one line: the samples, those classified as
// labelled (task accuracy), those predicting what the reference of
// prepare-reference.py predicts (the pass rate) and those that
// reference classifies as labelled (the accuracy of PyTorch on the same
// samples).
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
alignas(64) static float image[3 * 224 * 224];

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
  const auto *pixels = reinterpret_cast<const uint8_t *>(
      model::imageFile(MODEL_REGION, "pixels.data", &bytes));
  const auto *references = reinterpret_cast<const float *>(
      model::imageFile(MODEL_REGION, "reference.data", &bytes));
  const auto *labels = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_REGION, "labels.data", &bytes));
  const uint64_t samples = bytes / sizeof(int64_t);
  const float mean[] = {0.485f, 0.456f, 0.406f};
  const float deviation[] = {0.229f, 0.224f, 0.225f};
  uint64_t correct = 0, passed = 0, golden = 0;
  for (uint64_t sample = 0; sample < samples; ++sample) {
    // HWC pixels to the normalized NCHW image, in the order of the
    // reference.
    const uint8_t *source = pixels + sample * 224 * 224 * 3;
    for (unsigned i = 0; i < 224 * 224; ++i)
      for (unsigned c = 0; c < 3; ++c)
        image[c * 224 * 224 + i] =
            (source[i * 3 + c] / 255.0f - mean[c]) / deviation[c];
    Buffer<float, 4> input{
        image, image, 0, {1, 3, 224, 224}, {150528, 50176, 224, 1}};
    Buffer<float, 2> output;
    _mlir_ciface_forward(&output, &parameters, &bias, &quantized, &input);
    const float *got = output.aligned + output.offset;
    const float *want = references + sample * 1000;
    unsigned best = 0, gold = 0;
    for (unsigned i = 1; i < 1000; ++i) {
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

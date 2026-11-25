// The accuracy run of MobileNetV3-Small: the ImageNet validation images of
// the model region (a shard of the set; bench-device --bemu-accuracy packs it
// into the DDR image of each run), each normalized as the baremetal image
// is. Prints one line: the samples, those classified as labelled (task
// accuracy), those predicting what the FP32 model in PyTorch predicts (the
// pass rate) and those that model classifies as labelled (the accuracy of
// PyTorch on the same samples).
#include "ModelImage.h"
#include "model.h"
#include "testutils.h"

#include <stdint.h>
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
alignas(64) static float image[3 * 224 * 224];

int main() {
  const auto *p = reinterpret_cast<const float *>(parameters);
  const auto *b = reinterpret_cast<const int32_t *>(biases);
  const auto *w = reinterpret_cast<const int8_t *>(weights);
  mobilenet::Bias i32{b, b, 0, {sizeof(biases) / sizeof(int32_t)}, {1}};
  mobilenet::Weights i8{w, w, 0, {sizeof(weights)}, {1}};
  mobilenet::Parameters f32{p, p, 0, {sizeof(parameters) / 4}, {1}};
  // The samples of this run, in the model region.
  uint64_t bytes;
  const auto *pixels = reinterpret_cast<const uint8_t *>(
      model::imageFile(MODEL_IMAGE, "pixels.data", &bytes));
  const auto *references = reinterpret_cast<const float *>(
      model::imageFile(MODEL_IMAGE, "reference.data", &bytes));
  const auto *labels = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_IMAGE, "labels.data", &bytes));
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
    mobilenet::Input input{
        image, image, 0, {1, 3, 224, 224}, {150528, 50176, 224, 1}};
    mobilenet::Output output{};
    _mlir_ciface_forward(&output, &f32, &i32, &i8, &input);
    const float *values = output.data();
    const float *expected = references + sample * 1000;
    unsigned best = 0, gold = 0;
    for (unsigned i = 1; i < 1000; ++i) {
      if (values[i] > values[best])
        best = i;
      if (expected[i] > expected[gold])
        gold = i;
    }
    correct += best == labels[sample];
    golden += gold == labels[sample];
    // A sample passes when it predicts what the reference predicts.
    passed += best == gold;
    free(output.allocated);
  }
  printf(
      "Dataset: %lu samples, %lu correct, %lu passed, %lu reference "
      "correct\n",
      static_cast<unsigned long>(samples), static_cast<unsigned long>(correct),
      static_cast<unsigned long>(passed), static_cast<unsigned long>(golden));
  return 0;
}

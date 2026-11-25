// The accuracy run of LeNet: the MNIST test images of the model region (a
// shard of the test set; bench-device --bemu-accuracy packs it into the DDR
// image of each run). Prints one line: the
// samples, those classified as labelled (task accuracy), those predicting
// what the INT8 reference predicts (the pass rate) and
// those the reference classifies as labelled (the accuracy of PyTorch on the
// same samples).
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
int main() {
  const auto *p = reinterpret_cast<const float *>(parameters);
  const auto *b = reinterpret_cast<const int32_t *>(biases);
  const auto *w = reinterpret_cast<const int8_t *>(weights);
  lenet::Parameters f32{p, p, 0, {sizeof(parameters) / 4}, {1}};
  lenet::Bias i32{b, b, 0, {sizeof(biases) / 4}, {1}};
  lenet::Weights i8{w, w, 0, {sizeof(weights)}, {1}};
  // The samples of this run, in the model region.
  uint64_t bytes;
  const auto *images = reinterpret_cast<const float *>(
      model::imageFile(MODEL_IMAGE, "inputs.data", &bytes));
  const auto *references = reinterpret_cast<const float *>(
      model::imageFile(MODEL_IMAGE, "reference.data", &bytes));
  const auto *labels = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_IMAGE, "labels.data", &bytes));
  const uint64_t samples = bytes / sizeof(int64_t);
  uint64_t correct = 0, passed = 0, golden = 0;
  for (uint64_t sample = 0; sample < samples; ++sample) {
    const auto *image = images + sample * 28 * 28;
    lenet::Input input{image, image, 0, {1, 1, 28, 28}, {784, 784, 28, 1}};
    lenet::Output output{};
    _mlir_ciface_forward(&output, &f32, &i32, &i8, &input);
    const float *values = output.data();
    const float *expected = references + sample * 10;
    unsigned best = 0, gold = 0;
    for (unsigned i = 0; i < 10; ++i) {
      if (values[i] > values[best])
        best = i;
      if (expected[i] > expected[gold])
        gold = i;
    }
    const int64_t label = labels[sample];
    correct += best == label;
    golden += gold == label;
    // A sample passes when it predicts what the reference predicts: online
    // INT8 conversions in every layer make the logits' rounding noise grow
    // (the same reference in FP32 and FP64: half the BERT sentences beyond a
    // relative 0.02, yet 99.25% with the same class), so a bound on the
    // logits measures rounding, not correctness.
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

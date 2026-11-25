// BERT (emotion) compiled for Gemmini, the accuracy run: the validation
// sentences of the model region (a shard of the 2000; bench-device
// --bemu-accuracy packs it into the DDR image of each run). Prints one line:
// the samples, those classified as labelled (task accuracy), those
// predicting what the calibrated INT8 model in PyTorch predicts (the pass
// rate) and those that model classifies as labelled (the accuracy of
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

int main() {
  const auto *p = reinterpret_cast<const float *>(parameters);
  const auto *b = reinterpret_cast<const int32_t *>(biases);
  const auto *w = reinterpret_cast<const int8_t *>(weights);
  bert::Parameters f32{p, p, 0, {sizeof(parameters) / sizeof(float)}, {1}};
  bert::Bias i32{b, b, 0, {sizeof(biases) / sizeof(int32_t)}, {1}};
  bert::Weights i8{w, w, 0, {sizeof(weights)}, {1}};
  // The samples of this run, in the model region.
  uint64_t bytes;
  const auto *ids = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_REGION, "ids.data", &bytes));
  const auto *positions = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_REGION, "positions.data", &bytes));
  const auto *types = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_REGION, "types.data", &bytes));
  const auto *masks = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_REGION, "masks.data", &bytes));
  const auto *references = reinterpret_cast<const float *>(
      model::imageFile(MODEL_REGION, "reference.data", &bytes));
  const auto *labels = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_REGION, "labels.data", &bytes));
  const uint64_t samples = bytes / sizeof(int64_t);
  uint64_t correct = 0, passed = 0, golden = 0;
  for (uint64_t sample = 0; sample < samples; ++sample) {
    const int64_t *row[] = {ids + sample * 32, positions + sample * 32,
                            types + sample * 32, masks + sample * 32};
    bert::Tokens id{row[0], row[0], 0, {1, 32}, {32, 1}};
    bert::Tokens position{row[1], row[1], 0, {1, 32}, {32, 1}};
    bert::Tokens type{row[2], row[2], 0, {1, 32}, {32, 1}};
    bert::Tokens mask{row[3], row[3], 0, {1, 32}, {32, 1}};
    bert::Output output{};
    _mlir_ciface_forward(&output, &f32, &i32, &i8, &id, &position, &type,
                         &mask);
    const float *values = output.data();
    const float *expected = references + sample * 6;
    unsigned best = 0, gold = 0;
    for (unsigned i = 1; i < 6; ++i) {
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
  printf("Dataset: %lu samples, %lu correct, %lu passed, %lu reference "
         "correct\n",
         (unsigned long)samples, (unsigned long)correct, (unsigned long)passed,
         (unsigned long)golden);
  return 0;
}

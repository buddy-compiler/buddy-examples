// BERT (emotion) on the MAGIA device, the accuracy run: the validation
// sentences of the model region (a shard of the 2000; bench-device
// --bemu-accuracy packs it into the DDR image of each run). Prints one line:
// the samples, those classified as labelled (task accuracy), those
// predicting what the model in PyTorch with the mesh's products predicts
// (the pass rate) and those that model classifies as labelled (the accuracy
// of PyTorch on the same samples).
#include "ModelImage.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// A memref of the compiled graph.
template <typename T, int N> struct Buffer {
  const T *allocated;
  const T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};

extern "C" void
_mlir_ciface_forward(Buffer<float, 2> *output, Buffer<uint16_t, 1> *weights,
                     Buffer<float, 1> *parameters, Buffer<int64_t, 2> *ids,
                     Buffer<int64_t, 2> *positions, Buffer<int64_t, 2> *types,
                     Buffer<int64_t, 2> *mask);

// The weights of the Linears, which the mesh reads in place.
alignas(64) static const unsigned char weight_data[]
    __attribute__((section(".parameters"))) = {
#embed "arg0.data"
};
alignas(64) static const unsigned char parameter_data[] = {
#embed "arg1.data"
};

int main() {
  const auto *halves = reinterpret_cast<const uint16_t *>(weight_data);
  const auto *floats = reinterpret_cast<const float *>(parameter_data);
  Buffer<uint16_t, 1> weights{
      halves, halves, 0, {intptr_t(sizeof(weight_data) / 2)}, {1}};
  Buffer<float, 1> parameters{
      floats, floats, 0, {intptr_t(sizeof(parameter_data) / 4)}, {1}};
  // The samples of this run, in the model region.
  uint64_t bytes;
  const int64_t *rows[4];
  const char *names[] = {"ids.data", "positions.data", "types.data",
                         "masks.data"};
  for (unsigned i = 0; i < 4; ++i)
    rows[i] = reinterpret_cast<const int64_t *>(
        model::imageFile(MODEL_IMAGE, names[i], &bytes));
  const auto *references = reinterpret_cast<const float *>(
      model::imageFile(MODEL_IMAGE, "reference.data", &bytes));
  const auto *labels = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_IMAGE, "labels.data", &bytes));
  const uint64_t samples = bytes / sizeof(int64_t);
  uint64_t correct = 0, passed = 0, golden = 0;
  for (uint64_t sample = 0; sample < samples; ++sample) {
    Buffer<int64_t, 2> inputs[4];
    for (unsigned i = 0; i < 4; ++i) {
      const int64_t *row = rows[i] + sample * 32;
      inputs[i] = {row, row, 0, {1, 32}, {32, 1}};
    }
    Buffer<float, 2> output;
    _mlir_ciface_forward(&output, &weights, &parameters, &inputs[0], &inputs[1],
                         &inputs[2], &inputs[3]);
    const float *got = output.aligned + output.offset;
    const float *want = references + sample * 6;
    unsigned best = 0, gold = 0;
    for (unsigned i = 1; i < 6; ++i) {
      best = got[i] > got[best] ? i : best;
      gold = want[i] > want[gold] ? i : gold;
    }
    correct += best == labels[sample];
    golden += gold == labels[sample];
    // A sample passes when it predicts what the reference predicts.
    passed += best == gold;
    free(const_cast<float *>(output.allocated));
  }
  printf("Dataset: %lu samples, %lu correct, %lu passed, %lu reference "
         "correct\n",
         (unsigned long)samples, (unsigned long)correct, (unsigned long)passed,
         (unsigned long)golden);
  return 0;
}

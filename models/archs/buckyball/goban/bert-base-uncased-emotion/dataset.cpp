// The accuracy run of BERT emotion: the validation sentences of the model
// region (a shard of the set; bench-device --bemu-accuracy packs it into the
// DDR image of each run).
// Prints one line: the samples, those classified as labelled (task
// accuracy), those predicting what the INT8 reference predicts (the pass
// rate) and those the reference classifies as labelled (the accuracy of
// PyTorch on the same samples).
#include "ModelImage.h"
#include "model.h"
#include "testutils.h"

extern "C" {
#include "runtime.h"
}
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
  int64_t type_data[32] = {}, position_data[32];
  for (unsigned i = 0; i < 32; ++i)
    position_data[i] = i;
  bert::Parameters f32{p, p, 0, {sizeof(parameters) / sizeof(float)}, {1}};
  bert::Bias i32{b, b, 0, {sizeof(biases) / sizeof(int32_t)}, {1}};
  bert::Weights i8{w, w, 0, {sizeof(weights)}, {1}};
  bert::Tokens types{type_data, type_data, 0, {1, 32}, {32, 1}};
  bert::Tokens positions{position_data, position_data, 0, {1, 32}, {32, 1}};
  // The samples of this run, in the model region.
  uint64_t bytes;
  const auto *tokens = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_IMAGE, "ids.data", &bytes));
  const auto *masks = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_IMAGE, "masks.data", &bytes));
  const auto *references = reinterpret_cast<const float *>(
      model::imageFile(MODEL_IMAGE, "reference.data", &bytes));
  const auto *labels = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_IMAGE, "labels.data", &bytes));
  const uint64_t samples = bytes / sizeof(int64_t);
  uint64_t correct = 0, passed = 0, golden = 0;
  for (uint64_t sample = 0; sample < samples; ++sample) {
    const auto *ids_data = tokens + sample * 32;
    const auto *mask_values = masks + sample * 32;
    bert::Tokens ids{ids_data, ids_data, 0, {1, 32}, {32, 1}};
    bert::Tokens mask{mask_values, mask_values, 0, {1, 32}, {32, 1}};
    bert::Output output{};
    bert::Arguments arguments{&output, &f32,   &i32,  &i8,
                              &ids,    &types, &mask, &positions};
    struct runtime_hart hart =
        runtime_hart(0, DEVICE_HARTS_0_SIGNATURE, nullptr);
    struct runtime_job job{};
    job.signature = hart.signature;
    job.hart = 0;
    job.argument = &arguments;
    job.entry = [](void *, void *opaque) {
      auto &args = *static_cast<bert::Arguments *>(opaque);
      _mlir_ciface_forward(args.output, args.parameters, args.bias,
                           args.weights, args.ids, args.types, args.mask,
                           args.positions);
      return 0;
    };
    struct runtime_result result;
    struct runtime runtime;
    runtime_init(&runtime, &hart, 1, &job, 1, &result);
    runtime_baremetal_run(&runtime, 0);
    if (result.state != RUNTIME_SUCCEEDED || result.status)
      return 1;
    const float *values = output.data();
    const float *expected = references + sample * 6;
    unsigned best = 0, gold = 0;
    for (unsigned i = 0; i < 6; ++i) {
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

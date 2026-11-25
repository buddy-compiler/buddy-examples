// The accuracy run of MLP-Mixer: the ImageNet validation images of the model
// region (a shard of the set; bench-device --bemu-accuracy packs it into the
// DDR image of each run), each scaled to [-1, 1] and cut into the 196
// patches of the baremetal sample. Prints one line: the samples, those
// classified as labelled (task accuracy), those predicting what the INT8
// reference predicts (the pass rate) and those the reference classifies as
// labelled (the accuracy of PyTorch on the same samples).
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
alignas(64) static float patches[196 * 768];

int main() {
  const auto *p = reinterpret_cast<const float *>(parameters);
  const auto *b = reinterpret_cast<const int32_t *>(biases);
  const auto *w = reinterpret_cast<const int8_t *>(weights);
  mixer::Parameters f32{p, p, 0, {sizeof(parameters) / sizeof(float)}, {1}};
  mixer::Bias i32{b, b, 0, {sizeof(biases) / sizeof(int32_t)}, {1}};
  mixer::Weights i8{w, w, 0, {sizeof(weights)}, {1}};
  // The samples of this run, in the model region.
  uint64_t bytes;
  const auto *pixels = reinterpret_cast<const uint8_t *>(
      model::imageFile(MODEL_IMAGE, "pixels.data", &bytes));
  const auto *references = reinterpret_cast<const float *>(
      model::imageFile(MODEL_IMAGE, "reference.data", &bytes));
  const auto *labels = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_IMAGE, "labels.data", &bytes));
  const uint64_t samples = bytes / sizeof(int64_t);
  uint64_t correct = 0, passed = 0, golden = 0;
  for (uint64_t sample = 0; sample < samples; ++sample) {
    // Patch (row, column) of 16x16 pixels, its values channel by channel.
    const uint8_t *source = pixels + sample * 224 * 224 * 3;
    for (unsigned patch = 0; patch < 196; ++patch)
      for (unsigned c = 0; c < 3; ++c)
        for (unsigned y = 0; y < 16; ++y)
          for (unsigned x = 0; x < 16; ++x)
            patches[patch * 768 + c * 256 + y * 16 + x] =
                source[((patch / 14 * 16 + y) * 224 + patch % 14 * 16 + x) * 3 +
                       c] /
                    127.5f -
                1.0f;
    mixer::Input input{patches, patches, 0, {196, 768}, {768, 1}};
    mixer::Output output{};
    mixer::Arguments arguments{&output, &f32, &i32, &i8, &input};
    struct runtime_hart hart =
        runtime_hart(0, DEVICE_HARTS_0_SIGNATURE, nullptr);
    struct runtime_job job{};
    job.signature = hart.signature;
    job.hart = 0;
    job.argument = &arguments;
    job.entry = [](void *, void *opaque) {
      auto &args = *static_cast<mixer::Arguments *>(opaque);
      _mlir_ciface_forward(args.output, args.parameters, args.bias,
                           args.weights, args.input);
      return 0;
    };
    struct runtime_result result;
    struct runtime runtime;
    runtime_init(&runtime, &hart, 1, &job, 1, &result);
    runtime_baremetal_run(&runtime, 0);
    if (result.state != RUNTIME_SUCCEEDED || result.status)
      return 1;
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
    // A sample passes when it predicts what the reference predicts: the
    // logits' rounding noise grows through the online INT8 conversions.
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

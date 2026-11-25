// The accuracy run of Laya: the held-out typed decisions of the model region
// (a shard of them; bench-device --bemu-accuracy packs it beside the weights
// into the DDR image of each run), each as baremetal.cpp runs its one.
// Prints one line: the samples, those choosing the labelled option (task
// accuracy), those whose choice and action are the reference's (the pass
// rate) and those the reference chooses as labelled (the accuracy of PyTorch
// on the same samples).
#include "ModelImage.h"
#include "model.h"
#include "runtime.h"
#include "testutils.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  uint64_t bytes;
  const auto *p = reinterpret_cast<const float *>(
      model::imageFile(MODEL_IMAGE, "ddr/arg0.data", &bytes));
  laya::Parameters parameters{p, p, 0, {int64_t(bytes / 4)}, {1}};
  const auto *b = reinterpret_cast<const int32_t *>(
      model::imageFile(MODEL_IMAGE, "ddr/arg1.data", &bytes));
  laya::Bias bias{b, b, 0, {int64_t(bytes / 4)}, {1}};
  const auto *w = reinterpret_cast<const int8_t *>(
      model::imageFile(MODEL_IMAGE, "ddr/arg2.data", &bytes));
  laya::Weights weights{w, w, 0, {int64_t(bytes)}, {1}};
  // The samples of this run, in the model region.
  const auto *inputs = reinterpret_cast<const laya::Input *>(
      model::imageFile(MODEL_IMAGE, "inputs.data", &bytes));
  const auto *references = reinterpret_cast<const float *>(
      model::imageFile(MODEL_IMAGE, "reference.data", &bytes));
  const auto *labels = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_IMAGE, "labels.data", &bytes));
  const uint64_t samples = bytes / sizeof(int64_t);
  uint64_t correct = 0, passed = 0, golden = 0;
  for (uint64_t sample = 0; sample < samples; ++sample) {
    const laya::Input &input = inputs[sample];
    laya::Tokens tokens{input.tokens, input.tokens, 0, {1, 128}, {128, 1}};
    laya::Tokens mask{input.mask, input.mask, 0, {1, 128}, {128, 1}};
    laya::Tokens positions{
        input.positions, input.positions, 0, {1, 16}, {16, 1}};
    laya::Valid valid{input.valid, input.valid, 0, {1, 16}, {16, 1}};
    laya::Type type{input.type, input.type, 0, {1}, {1}};
    laya::Output output{};
    laya::Arguments arguments{&output, &parameters, &bias,  &weights, &tokens,
                              &mask,   &positions,  &valid, &type};
    struct runtime_hart hart =
        runtime_hart(0, DEVICE_HARTS_0_SIGNATURE, nullptr);
    struct runtime_job job{};
    job.signature = hart.signature;
    job.hart = 0;
    job.argument = &arguments;
    job.entry = [](void *, void *opaque) {
      auto &a = *static_cast<laya::Arguments *>(opaque);
      _mlir_ciface_forward(a.output, a.parameters, a.bias, a.weights, a.tokens,
                           a.mask, a.positions, a.valid, a.type);
      return 0;
    };
    struct runtime_result result;
    struct runtime runtime;
    runtime_init(&runtime, &hart, 1, &job, 1, &result);
    runtime_baremetal_run(&runtime, 0);
    if (result.state != RUNTIME_SUCCEEDED || result.status)
      return 1;
    const float *expected = references + sample * 18;
    const float *scores = output.choice.data();
    unsigned choice = 0, gold = 0;
    for (unsigned i = 1; i < 16; ++i) {
      if (scores[i] > scores[choice])
        choice = i;
      if (expected[i] > expected[gold])
        gold = i;
    }
    const unsigned action = output.action.data()[1] > output.action.data()[0];
    const unsigned goldAction = expected[17] > expected[16];
    correct += choice == labels[sample];
    golden += gold == labels[sample];
    // A sample passes when its choice and action are the reference's: the
    // scores' rounding noise grows through the online INT8 conversions.
    passed += choice == gold && action == goldAction;
    free(output.choice.allocated);
    free(output.action.allocated);
  }
  printf(
      "Dataset: %lu samples, %lu correct, %lu passed, %lu reference "
      "correct\n",
      static_cast<unsigned long>(samples), static_cast<unsigned long>(correct),
      static_cast<unsigned long>(passed), static_cast<unsigned long>(golden));
  return 0;
}

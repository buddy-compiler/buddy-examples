#include "ModelImage.h"
#include "model.h"
#include "runtime.h"
#include "testutils.h"

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
  const auto &input = *reinterpret_cast<const laya::Input *>(
      model::imageFile(MODEL_IMAGE, "input.data", &bytes));
  laya::Tokens tokens{input.tokens, input.tokens, 0, {1, 128}, {128, 1}};
  laya::Tokens mask{input.mask, input.mask, 0, {1, 128}, {128, 1}};
  laya::Tokens positions{input.positions, input.positions, 0, {1, 16}, {16, 1}};
  laya::Valid valid{input.valid, input.valid, 0, {1, 16}, {16, 1}};
  laya::Type type{input.type, input.type, 0, {1}, {1}};
  laya::Output output{};
  laya::Arguments arguments{&output, &parameters, &bias,  &weights, &tokens,
                            &mask,   &positions,  &valid, &type};
  struct runtime_core core = runtime_core(0, DEVICE_CORES_0_SIGNATURE, nullptr);
  struct runtime_job job{};
  job.signature = core.signature;
  job.core = 0;
  job.argument = &arguments;
  job.entry = [](void *, void *opaque) {
    auto &a = *static_cast<laya::Arguments *>(opaque);
    _mlir_ciface_forward(a.output, a.parameters, a.bias, a.weights, a.tokens,
                         a.mask, a.positions, a.valid, a.type);
    return 0;
  };
  struct runtime_result result;
  struct runtime runtime;
  runtime_init(&runtime, &core, 1, &job, 1, &result);
  const uint64_t begin = read_counter();
  runtime_baremetal_run(&runtime, 0);
  const uint64_t elapsed = read_counter() - begin;
  if (result.state != RUNTIME_SUCCEEDED || result.status)
    return 1;
  const auto *expected = reinterpret_cast<const float *>(
      model::imageFile(MODEL_IMAGE, "expected.data", &bytes));
  unsigned choice = 0, action = 0, gold_choice = 0, gold_action = 0;
  for (unsigned i = 1; i < 16; ++i) {
    if (output.choice.data()[i] > output.choice.data()[choice])
      choice = i;
    if (expected[i] > expected[gold_choice])
      gold_choice = i;
  }
  for (int64_t i = 1; i < output.action.sizes[1]; ++i) {
    if (output.action.data()[i] > output.action.data()[action])
      action = i;
    if (expected[16 + i] > expected[16 + gold_action])
      gold_action = i;
  }
  printf("Choice: %u\nAction: %u\nInference cycles: %lu\n", choice, action,
         static_cast<unsigned long>(elapsed));
  free(output.choice.allocated);
  free(output.action.allocated);
  return choice != gold_choice || action != gold_action;
}

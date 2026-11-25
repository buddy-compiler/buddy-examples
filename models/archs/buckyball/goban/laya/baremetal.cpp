#include "ModelImage.h"
#include "model.h"
#include "runtime.h"
#include "testutils.h"
#include <initializer_list>

// Stream the final CPU-visible scores for external simulation checks.
static const uint32_t reference[18]
    __attribute__((section(".bbsim.reference"), used)) = {};
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} capture __attribute__((section(".bbsim.capture_output"), used, retain)) = {
    nullptr, reference, sizeof(reference)};

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
  struct runtime_hart hart = runtime_hart(0, DEVICE_HARTS_0_SIGNATURE, nullptr);
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
  const uint64_t begin = read_counter();
  runtime_baremetal_run(&runtime, 0);
  const uint64_t elapsed = read_counter() - begin;
  if (result.state != RUNTIME_SUCCEEDED || result.status)
    return 1;
  // The choice and action scores against the INT8 reference of the recipe
  // (expected.data, frontend/reference.py).
  const auto *expected = reinterpret_cast<const float *>(
      model::imageFile(MODEL_IMAGE, "expected.data", &bytes));
  float scores[18];
  __builtin_memcpy(scores, output.choice.data(), 16 * sizeof(float));
  __builtin_memcpy(scores + 16, output.action.data(), 2 * sizeof(float));
  unsigned choice = 0, gold_choice = 0;
  for (unsigned i = 1; i < 16; ++i) {
    if (scores[i] > scores[choice])
      choice = i;
    if (expected[i] > expected[gold_choice])
      gold_choice = i;
  }
  const unsigned action = scores[17] > scores[16];
  const unsigned gold_action = expected[17] > expected[16];
  printf("Choice: %u, reference %u\nAction: %u, reference %u\n"
         "Inference cycles: %lu\n",
         choice, gold_choice, action, gold_action,
         static_cast<unsigned long>(elapsed));
  // The reference computes the device's INT8 model; FP32 orders of norms and
  // softmax may still flip a code. The tolerance is 2.5 INT8 steps (1/127) of
  // the largest reference score.
  const int failed = check_output(scores, expected, 18, 2e-2f) ||
                     choice != gold_choice || action != gold_action;
  uintptr_t owner;
  asm volatile("csrr %0, mhartid" : "=r"(owner));
  auto stream =
      reinterpret_cast<volatile uint32_t *>(0x60000000ULL + owner * 0x40000);
  *stream = 0x42424f55;
  for (const auto *scores : {output.choice.data(), output.action.data()}) {
    const unsigned count = scores == output.choice.data() ? 16 : 2;
    for (unsigned i = 0; i < count; ++i) {
      uint32_t word;
      __builtin_memcpy(&word, scores + i, sizeof(word));
      *stream = word;
    }
  }
  free(output.choice.allocated);
  free(output.action.allocated);
  return failed;
}

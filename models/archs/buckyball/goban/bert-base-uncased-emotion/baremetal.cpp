#include "model.h"
#include "testutils.h"

extern "C" {
#include "runtime.h"
}
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
alignas(64) static const unsigned char tokens[] = {
#embed "input_ids.data"
};
alignas(64) static const unsigned char mask_data[] = {
#embed "attention_mask.data"
};
static_assert(sizeof(tokens) == 32 * sizeof(int64_t));
static_assert(sizeof(mask_data) == 32 * sizeof(int64_t));
// The logits of the INT8 reference model of the recipe on that sentence.
alignas(64) static const unsigned char golden[] = {
#embed "reference.data"
};
static_assert(sizeof(golden) == 6 * sizeof(float));

// Stream the final CPU-visible scores for external simulation checks.
static const uint32_t reference[6]
    __attribute__((section(".bbsim.reference"), used)) = {};
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} capture __attribute__((section(".bbsim.capture_output"), used, retain)) = {
    nullptr, reference, sizeof(reference)};

int main() {
  const auto *p = reinterpret_cast<const float *>(parameters);
  const auto *b = reinterpret_cast<const int32_t *>(biases);
  const auto *w = reinterpret_cast<const int8_t *>(weights);
  const auto *ids_data = reinterpret_cast<const int64_t *>(tokens);
  const auto *mask_values = reinterpret_cast<const int64_t *>(mask_data);
  int64_t type_data[32] = {}, position_data[32];
  for (unsigned i = 0; i < 32; ++i)
    position_data[i] = i;
  bert::Parameters f32{p, p, 0, {sizeof(parameters) / sizeof(float)}, {1}};
  bert::Bias i32{b, b, 0, {sizeof(biases) / sizeof(int32_t)}, {1}};
  bert::Weights i8{w, w, 0, {sizeof(weights)}, {1}};
  bert::Tokens ids{ids_data, ids_data, 0, {1, 32}, {32, 1}};
  bert::Tokens types{type_data, type_data, 0, {1, 32}, {32, 1}};
  bert::Tokens mask{mask_values, mask_values, 0, {1, 32}, {32, 1}};
  bert::Tokens positions{position_data, position_data, 0, {1, 32}, {32, 1}};
  bert::Output output{};
  bert::Arguments arguments{&output, &f32,   &i32,  &i8,
                            &ids,    &types, &mask, &positions};
  struct runtime_hart hart = runtime_hart(0, DEVICE_HARTS_0_SIGNATURE, nullptr);
  struct runtime_job job{};
  job.signature = hart.signature;
  job.hart = 0;
  job.argument = &arguments;
  job.entry = [](void *, void *opaque) {
    auto &args = *static_cast<bert::Arguments *>(opaque);
    _mlir_ciface_forward(args.output, args.parameters, args.bias, args.weights,
                         args.ids, args.types, args.mask, args.positions);
    return 0;
  };
  struct runtime_result result;
  struct runtime runtime;
  runtime_init(&runtime, &hart, 1, &job, 1, &result);
  const uint64_t start = read_counter();
  runtime_baremetal_run(&runtime, 0);
  const uint64_t elapsed = read_counter() - start;
  if (result.state != RUNTIME_SUCCEEDED || result.status)
    return 1;
  const float *values = output.data();
  const auto *expected = reinterpret_cast<const float *>(golden);
  unsigned best = 0, gold = 0;
  for (unsigned i = 1; i < 6; ++i) {
    if (values[i] > values[best])
      best = i;
    if (expected[i] > expected[gold])
      gold = i;
  }
  const char *labels[] = {"sadness", "joy",  "love",
                          "anger",   "fear", "surprise"};
  printf("Classification: %s, reference %s\nInference cycles: %lu\n",
         labels[best], labels[gold], static_cast<unsigned long>(elapsed));
  // The reference computes the device's INT8 model; FP32 orders of norms and
  // softmax may still flip a code. The tolerance is 2.5 INT8 steps (1/127) of
  // the largest reference score.
  const int failed = check_output(values, expected, 6, 2e-2f) || best != gold;
  uintptr_t owner;
  asm volatile("csrr %0, mhartid" : "=r"(owner));
  auto stream =
      reinterpret_cast<volatile uint32_t *>(0x60000000ULL + owner * 0x40000);
  *stream = 0x42424f55;
  for (unsigned i = 0; i < 6; ++i) {
    uint32_t word;
    __builtin_memcpy(&word, values + i, sizeof(word));
    *stream = word;
  }
  free(output.allocated);
  return failed;
}

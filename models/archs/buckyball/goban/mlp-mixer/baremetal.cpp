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
alignas(64) static const unsigned char pixels[] = {
#embed "input.f32"
};
static_assert(sizeof(pixels) == 196 * 768 * sizeof(float));
// The logits of the INT8 reference model of the recipe on that input.
alignas(64) static const unsigned char golden[] = {
#embed "reference.data"
};
static_assert(sizeof(golden) == 1000 * sizeof(float));

// Stream the final CPU-visible scores for external simulation checks.
static const uint32_t reference[1000]
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
  const auto *image = reinterpret_cast<const float *>(pixels);
  mixer::Parameters f32{p, p, 0, {sizeof(parameters) / sizeof(float)}, {1}};
  mixer::Bias i32{b, b, 0, {sizeof(biases) / sizeof(int32_t)}, {1}};
  mixer::Weights i8{w, w, 0, {sizeof(weights)}, {1}};
  mixer::Input input{image, image, 0, {196, 768}, {768, 1}};
  mixer::Output output{};
  mixer::Arguments arguments{&output, &f32, &i32, &i8, &input};
  struct runtime_hart hart = runtime_hart(0, DEVICE_HARTS_0_SIGNATURE, nullptr);
  struct runtime_job job{};
  job.signature = hart.signature;
  job.hart = 0;
  job.argument = &arguments;
  job.entry = [](void *, void *opaque) {
    auto &args = *static_cast<mixer::Arguments *>(opaque);
    _mlir_ciface_forward(args.output, args.parameters, args.bias, args.weights,
                         args.input);
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
  for (unsigned i = 1; i < 1000; ++i) {
    if (values[i] > values[best])
      best = i;
    if (expected[i] > expected[gold])
      gold = i;
  }
  printf("Classification Index: %u, reference %u\nInference cycles: %lu\n",
         best, gold, static_cast<unsigned long>(elapsed));
  // The reference computes the device's INT8 model, but every block converts
  // to INT8 online: a different summation order or norm/GELU rounding flips
  // some codes and the flips grow through the eight blocks. The same reference
  // in FP64 instead of FP32 is 0.12 (relative) from it, so the bound is 0.15;
  // the classification must match.
  const int failed =
      check_output(values, expected, 1000, 0.15f) || best != gold;
  uintptr_t owner;
  asm volatile("csrr %0, mhartid" : "=r"(owner));
  auto stream =
      reinterpret_cast<volatile uint32_t *>(0x60000000ULL + owner * 0x40000);
  *stream = 0x42424f55;
  for (unsigned i = 0; i < 1000; ++i) {
    uint32_t word;
    __builtin_memcpy(&word, values + i, sizeof(word));
    *stream = word;
  }
  free(output.allocated);
  return failed;
}

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
  struct runtime_core core = runtime_core(0, DEVICE_CORES_0_SIGNATURE, nullptr);
  struct runtime_job job{};
  job.signature = core.signature;
  job.core = 0;
  job.argument = &arguments;
  job.entry = [](void *, void *opaque) {
    auto &args = *static_cast<mixer::Arguments *>(opaque);
    _mlir_ciface_forward(args.output, args.parameters, args.bias, args.weights,
                         args.input);
    return 0;
  };
  struct runtime_result result;
  struct runtime runtime;
  runtime_init(&runtime, &core, 1, &job, 1, &result);
  const uint64_t start = read_counter();
  runtime_baremetal_run(&runtime, 0);
  const uint64_t elapsed = read_counter() - start;
  if (result.state != RUNTIME_SUCCEEDED || result.status)
    return 1;
  const float *values = output.data();
  unsigned best = 0;
  for (unsigned i = 1; i < 1000; ++i)
    if (values[i] > values[best])
      best = i;
  printf("Classification Index: %u\nInference cycles: %lu\n", best,
         static_cast<unsigned long>(elapsed));
  free(output.allocated);
  return 0;
}

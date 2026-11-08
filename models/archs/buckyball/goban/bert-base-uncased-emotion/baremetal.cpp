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
  struct runtime_core core = runtime_core(0, DEVICE_CORES_0_SIGNATURE, nullptr);
  struct runtime_job job{};
  job.signature = core.signature;
  job.core = 0;
  job.argument = &arguments;
  job.entry = [](void *, void *opaque) {
    auto &args = *static_cast<bert::Arguments *>(opaque);
    _mlir_ciface_forward(args.output, args.parameters, args.bias, args.weights,
                         args.ids, args.types, args.mask, args.positions);
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
  for (unsigned i = 1; i < 6; ++i)
    if (values[i] > values[best])
      best = i;
  const char *labels[] = {"sadness", "joy",  "love",
                          "anger",   "fear", "surprise"};
  printf("Classification: %s\nInference cycles: %lu\n", labels[best],
         static_cast<unsigned long>(elapsed));
  free(output.allocated);
  return 0;
}

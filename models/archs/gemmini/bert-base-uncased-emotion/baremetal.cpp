#include "model.h"
#include "testutils.h"

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
#embed "inputs.data"
};
static_assert(sizeof(tokens) == 4 * 32 * sizeof(int64_t));
// The logits of the INT8 reference model of the recipe on that sentence.
alignas(64) static const unsigned char golden[] = {
#embed "reference.data"
};
static_assert(sizeof(golden) == 6 * sizeof(float));

int main() {
  const auto *p = reinterpret_cast<const float *>(parameters);
  const auto *b = reinterpret_cast<const int32_t *>(biases);
  const auto *w = reinterpret_cast<const int8_t *>(weights);
  const auto *ids_data = reinterpret_cast<const int64_t *>(tokens);
  bert::Parameters f32{p, p, 0, {sizeof(parameters) / sizeof(float)}, {1}};
  bert::Bias i32{b, b, 0, {sizeof(biases) / sizeof(int32_t)}, {1}};
  bert::Weights i8{w, w, 0, {sizeof(weights)}, {1}};
  bert::Tokens ids{ids_data, ids_data, 0, {1, 32}, {32, 1}};
  bert::Tokens types{ids_data + 64, ids_data + 64, 0, {1, 32}, {32, 1}};
  bert::Tokens mask{ids_data + 96, ids_data + 96, 0, {1, 32}, {32, 1}};
  bert::Tokens positions{ids_data + 32, ids_data + 32, 0, {1, 32}, {32, 1}};
  bert::Output output{};
  const auto start = read_counter();
  _mlir_ciface_forward(&output, &f32, &i32, &i8, &ids, &positions, &types,
                       &mask);
  const auto elapsed = read_counter() - start;
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
  free(output.allocated);
  return failed;
}

#include "checked.h"
// Qwen3-0.6B on the MAGIA device: Loom runs the compiled graph bare metal,
// the mesh its matrix products. The prompt, then one token a step through
// the KV cache, against the tokens the model writes in PyTorch with the
// mesh's products in software (buddy-pytorch-importer.py).
#include "testutils.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

constexpr unsigned layers = 28;
constexpr unsigned heads = 8, capacity = 256, width = 128;
constexpr unsigned vocabulary = 151936;

// A memref of the compiled graph.
template <typename T, int N> struct Buffer {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};

// What a step returns: the logits of its tokens and the updated cache, every
// buffer owned by the caller.
struct Results {
  Buffer<float, 3> logits;
  Buffer<float, 4> keys[layers];
  Buffer<float, 4> values[layers];
};

extern "C" void _mlir_ciface_forward();

// The entry takes the FP16 weights and the FP32 parameters, then the tokens,
// the first position and the cache of every layer in the order of
// input-order.txt: 2 + 2 * layers buffers.
template <unsigned... I>
static void forward(Results *results, Buffer<uint16_t, 1> *weights,
                    Buffer<float, 1> *parameters, void **inputs) {
  using Entry = void (*)(Results *, Buffer<uint16_t, 1> *, Buffer<float, 1> *,
                         decltype((void)I, (void *)nullptr)...);
  reinterpret_cast<Entry>(&_mlir_ciface_forward)(results, weights, parameters,
                                                 inputs[I]...);
}

static const char order_text[] = {
#embed "input-order.txt"
    , 0};
alignas(8) static const unsigned char token_data[] = {
#embed "check-tokens.data"
};
// The reference logits of the last position of every step.
alignas(8) static const unsigned char logit_data[] = {
#embed "check-logits.data"
};
constexpr unsigned steps = 8;
static_assert(sizeof(token_data) == steps * sizeof(int64_t));
static_assert(sizeof(logit_data) == steps * vocabulary * sizeof(float));

// Loom's stack is small: what a step passes lies here.
static Buffer<float, 4> cache[2 * layers];
static void *example[2 + 2 * layers];
static void *inputs[2 + 2 * layers];
static Results results;
static int64_t tokens[capacity];
static float outputs[steps * vocabulary];

int check_model(const uint16_t *weight_data, intptr_t weight_count,
                const float *parameter_data, intptr_t parameter_count,
                const int64_t *input, unsigned input_count) {
  auto *halves = const_cast<uint16_t *>(weight_data);
  auto *floats = const_cast<float *>(parameter_data);
  const auto *prompt = input;
  const auto *expected = reinterpret_cast<const int64_t *>(token_data);
  unsigned count = input_count;
  if (!count || count + steps > capacity)
    return 1;
  for (unsigned i = 0; i < count; ++i)
    tokens[i] = prompt[i];
  Buffer<uint16_t, 1> weights{halves, halves, 0, {weight_count}, {1}};
  Buffer<float, 1> parameters{floats, floats, 0, {parameter_count}, {1}};
  // The cache: the keys of every layer, then the values, empty.
  for (auto &buffer : cache) {
    auto *data =
        static_cast<float *>(calloc(heads * capacity * width, sizeof(float)));
    if (!data)
      abort();
    buffer = {data,
              data,
              0,
              {1, heads, capacity, width},
              {heads * capacity * width, capacity * width, width, 1}};
  }

  int64_t start = 0;
  unsigned same = 0;
  uint64_t total = 0;
  for (unsigned step = 0; step < steps; ++step) {
    Buffer<int64_t, 2> ids{
        tokens, tokens, 0, {1, intptr_t(count)}, {intptr_t(count), 1}};
    Buffer<int64_t, 1> first{&start, &start, 0, {1}, {1}};
    example[0] = &ids;
    example[1] = &first;
    for (unsigned index = 0; index < 2 * layers; ++index)
      example[2 + index] = &cache[index];
    const char *cursor = order_text;
    for (unsigned index = 0; index < 2 + 2 * layers; ++index)
      inputs[index] =
          example[strtoul(cursor, const_cast<char **>(&cursor), 10)];
    const uint64_t begin = read_counter();
    forward<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
            53, 54, 55, 56, 57>(&results, &weights, &parameters, inputs);
    const uint64_t elapsed = read_counter() - begin;
    total += elapsed;
    // The next token: the largest logit of the last position.
    const float *last = results.logits.aligned + results.logits.offset +
                        (count - 1) * vocabulary;
    unsigned token = 0;
    for (unsigned i = 1; i < vocabulary; ++i)
      if (last[i] > last[token])
        token = i;
    for (unsigned i = 0; i < vocabulary; ++i)
      outputs[step * vocabulary + i] = last[i];
    free(results.logits.allocated);
    for (unsigned layer = 0; layer < layers; ++layer) {
      free(cache[layer].allocated);
      free(cache[layers + layer].allocated);
      cache[layer] = results.keys[layer];
      cache[layers + layer] = results.values[layer];
    }
    start += count;
    tokens[0] = token;
    count = 1;
    same += token == expected[step];
    printf("Token %u: %u (the reference %u), %lu cycles\n", step, token,
           static_cast<unsigned>(expected[step]),
           static_cast<unsigned long>(elapsed));
  }
  printf("Inference cycles: %lu\n", static_cast<unsigned long>(total));
  // The reference computes the mesh's FP16 products the same way, but the
  // other operations round differently in FP32, which flips FP16 roundings
  // of the inputs of later products over the 28 layers: the compiled model
  // on the host (qwen3-0-6b-model-check) stays within 3.5% of the largest
  // logit of every step, with every token the same.
  const int failed =
      check_output(outputs, reinterpret_cast<const float *>(logit_data),
                   steps * vocabulary, 5e-2f);
  printf("Tokens: %u/%u as the reference\n", same, steps);
  for (auto &entry : cache)
    free(entry.allocated);
  return failed || same != steps;
}

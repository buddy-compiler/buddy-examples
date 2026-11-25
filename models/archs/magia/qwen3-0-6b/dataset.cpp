// Qwen3-0.6B on the MAGIA device, the accuracy run: the LAMBADA passages of
// the model region (a shard of them; bench-device --bemu-accuracy packs it
// beside the weights into the DDR image of each run). Each passage without
// its last word is written into an empty KV cache, then the last word's
// tokens one by one (given, as checked.cpp decodes); at each the program's
// next token is the largest logit of the last position. Prints one line:
// the samples, those predicting every token of the last word (task
// accuracy), those predicting what the FP32 checkpoint in PyTorch predicts
// at every position (the pass rate) and those the checkpoint predicts right
// (its accuracy on the same samples).
#include "ModelImage.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// Loom's stack is small: what a step passes lies here.
static Buffer<float, 4> cache[2 * layers];
static void *example[2 + 2 * layers];
static void *inputs[2 + 2 * layers];
static Results results;
static int64_t tokens[capacity];

int main() {
  uint64_t bytes;
  auto *halves = reinterpret_cast<uint16_t *>(
      const_cast<char *>(model::imageFile(MODEL_BASE, "arg0.data", &bytes)));
  Buffer<uint16_t, 1> weights{
      halves, halves, 0, {intptr_t(bytes / sizeof(uint16_t))}, {1}};
  auto *floats = reinterpret_cast<float *>(
      const_cast<char *>(model::imageFile(MODEL_BASE, "arg1.data", &bytes)));
  Buffer<float, 1> parameters{
      floats, floats, 0, {intptr_t(bytes / sizeof(float))}, {1}};
  const auto *contexts = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_BASE, "contexts.data", &bytes));
  const auto *targets = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_BASE, "targets.data", &bytes));
  const auto *reference = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_BASE, "reference.data", &bytes));
  const auto *lengths = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_BASE, "lengths.data", &bytes));
  const uint64_t samples = bytes / (2 * sizeof(int64_t));
  uint64_t correct = 0, passed = 0, golden = 0;
  for (uint64_t sample = 0; sample < samples; ++sample) {
    const int64_t used = lengths[2 * sample], count = lengths[2 * sample + 1];
    const int64_t *target = targets + sample * 8;
    const int64_t *expected = reference + sample * 8;
    memcpy(tokens, contexts + sample * 248, used * sizeof(int64_t));
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
    bool right = true, same = true, checkpoint = true;
    int64_t start = 0, size = used;
    for (int64_t step = 0; step < count; ++step) {
      Buffer<int64_t, 2> ids{tokens, tokens, 0, {1, size}, {size, 1}};
      Buffer<int64_t, 1> first{&start, &start, 0, {1}, {1}};
      example[0] = &ids;
      example[1] = &first;
      for (unsigned index = 0; index < 2 * layers; ++index)
        example[2 + index] = &cache[index];
      const char *cursor = order_text;
      for (unsigned index = 0; index < 2 + 2 * layers; ++index)
        inputs[index] =
            example[strtoul(cursor, const_cast<char **>(&cursor), 10)];
      forward<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
              19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
              35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
              51, 52, 53, 54, 55, 56, 57>(&results, &weights, &parameters,
                                          inputs);
      const float *last = results.logits.aligned + results.logits.offset +
                          (size - 1) * vocabulary;
      unsigned token = 0;
      for (unsigned i = 1; i < vocabulary; ++i)
        if (last[i] > last[token])
          token = i;
      free(results.logits.allocated);
      for (unsigned layer = 0; layer < layers; ++layer) {
        free(cache[layer].allocated);
        free(cache[layers + layer].allocated);
        cache[layer] = results.keys[layer];
        cache[layers + layer] = results.values[layer];
      }
      right = right && token == target[step];
      same = same && token == expected[step];
      checkpoint = checkpoint && expected[step] == target[step];
      // The next step reads the given token of the word.
      start += size;
      tokens[0] = target[step];
      size = 1;
    }
    for (auto &entry : cache)
      free(entry.allocated);
    correct += right;
    passed += same;
    golden += checkpoint;
  }
  printf("Dataset: %lu samples, %lu correct, %lu passed, %lu reference "
         "correct\n",
         (unsigned long)samples, (unsigned long)correct, (unsigned long)passed,
         (unsigned long)golden);
  return 0;
}

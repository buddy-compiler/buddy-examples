// The accuracy run of Qwen3-0.6B: the LAMBADA passages of the model region
// (a shard of them; bench-device --bemu-accuracy packs it beside the packs
// into the DDR image of each run). Each passage without its last word is
// written into an empty KV cache, then the last word's tokens one by one
// (given, as runner.cpp decodes); at each the program's next token is the
// first largest candidate. Prints one line: the samples, those predicting
// every token of the last word (task accuracy), those predicting what the
// FP32 checkpoint in PyTorch predicts at every position (the pass rate) and
// those the checkpoint predicts right (its accuracy on the same samples).
#include "testutils.h"
#include <device.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The program, lowered by lower-device: its words are the address of every
// buffer followed by its dynamic sizes, the outputs last.
extern "C" int program_job(void *hart, void *argument);

constexpr unsigned layers = 28, heads = 8, dimension = 128, capacity = 256;

// The address of the file `name` of the model image, and its bytes.
static const char *file(const char *name, uint64_t *bytes) {
  const char *line = strchr((const char *)MODEL_IMAGE, '\n') + 1;
  const size_t length = strlen(name);
  while (*line != '\n') {
    const char *end = strchr(line, '\n');
    if (!strncmp(line, name, length) && line[length] == '\t') {
      char *next;
      const uint64_t offset = strtoull(line + length + 1, &next, 10);
      *bytes = strtoull(next + 1, NULL, 10);
      return (const char *)MODEL_IMAGE + offset;
    }
    line = end + 1;
  }
  printf("%s: not in the model image\n", name);
  exit(1);
}

static void *words[3 + 4 * 128];

int main(void) {
  uint64_t bytes;
  const int64_t *lengths = (const int64_t *)file("lengths.data", &bytes);
  const uint64_t samples = bytes / (2 * sizeof(int64_t));
  uintptr_t hartid;
  asm volatile("csrr %0, mhartid" : "=r"(hartid));
  if (hartid != DEVICE_CONTROL_HART) {
    // One program run per target token of every passage.
    for (uint64_t sample = 0; sample < samples; ++sample)
      for (int64_t step = 0; step < lengths[2 * sample + 1]; ++step)
        program_run(program_job, NULL);
    return 0;
  }
  const char *pack_f32 = file("pack-f32.data", &bytes);
  const char *pack_i8 = file("pack-i8.data", &bytes);
  const char *pack_i32 = file("pack-i32.data", &bytes);
  const int64_t *contexts = (const int64_t *)file("contexts.data", &bytes);
  const int64_t *targets = (const int64_t *)file("targets.data", &bytes);
  const int64_t *reference = (const int64_t *)file("reference.data", &bytes);
  const char *text = file("input-order.txt", &bytes);
  unsigned order[2 + 2 * layers];
  for (unsigned i = 0; i < 2 + 2 * layers; ++i) {
    char *next;
    order[i] = strtoul(text, &next, 10);
    text = next;
  }
  text = file("candidates.txt", &bytes);
  char *next;
  strtoul(text, &next, 10);
  const uint64_t total = strtoul(next, NULL, 10);
  // The buffers after the model image: the KV caches (in and out) at the
  // largest capacity, and the candidates.
  char *scratch =
      (char *)MODEL_IMAGE +
      (strtoull((const char *)MODEL_IMAGE + 9, NULL, 10) + 4095) / 4096 * 4096;
  const uint64_t cacheBytes = heads * capacity * dimension * sizeof(float);
  float *cache[2 * layers], *updated[2 * layers];
  for (unsigned i = 0; i < 2 * layers; ++i) {
    cache[i] = (float *)scratch;
    updated[i] = (float *)(scratch + cacheBytes);
    scratch += 2 * cacheBytes;
  }
  float *values = (float *)scratch;
  int64_t *indices =
      (int64_t *)(scratch + (total * sizeof(float) + 63) / 64 * 64);
  uint64_t correct = 0, passed = 0, golden = 0;
  for (uint64_t sample = 0; sample < samples; ++sample) {
    const int64_t used = lengths[2 * sample], count = lengths[2 * sample + 1];
    const int64_t *context = contexts + sample * 248;
    const int64_t *target = targets + sample * 8;
    const int64_t *expected = reference + sample * 8;
    const uint64_t cacheCapacity = (used + count + 31) / 32 * 32;
    for (unsigned i = 0; i < 2 * layers; ++i)
      memset(cache[i], 0, heads * cacheCapacity * dimension * sizeof(float));
    bool right = true, same = true, checkpoint = true;
    int64_t position = 0, given = 0;
    for (int64_t step = 0; step < count; ++step) {
      // The passage first, then each target token but the last.
      const int64_t *ids = step ? &given : context;
      const uint64_t tokens = step ? 1 : used;
      void *inputs[2 + 2 * layers];
      inputs[0] = (void *)ids;
      inputs[1] = &position;
      for (unsigned i = 0; i < 2 * layers; ++i)
        inputs[2 + i] = cache[i];
      unsigned word = 0;
      words[word++] = (void *)pack_f32;
      words[word++] = (void *)pack_i8;
      words[word++] = (void *)pack_i32;
      for (unsigned index : order) {
        words[word++] = inputs[index];
        if (index == 0)
          words[word++] = (void *)(uintptr_t)tokens;
        else if (index >= 2)
          words[word++] = (void *)(uintptr_t)cacheCapacity;
      }
      words[word++] = values;
      words[word++] = indices;
      for (unsigned i = 0; i < 2 * layers; ++i) {
        words[word++] = updated[i];
        words[word++] = (void *)(uintptr_t)cacheCapacity;
      }
      if (program_run(program_job, words))
        return 1;
      for (unsigned i = 0; i < 2 * layers; ++i) {
        float *swap = cache[i];
        cache[i] = updated[i];
        updated[i] = swap;
      }
      unsigned best = 0;
      for (unsigned i = 1; i < total; ++i)
        if (values[i] > values[best])
          best = i;
      right = right && indices[best] == target[step];
      same = same && indices[best] == expected[step];
      checkpoint = checkpoint && expected[step] == target[step];
      position += tokens;
      given = target[step];
    }
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

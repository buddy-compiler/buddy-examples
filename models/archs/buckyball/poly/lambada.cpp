// Score the last word of each LAMBADA passage with the compiled Poly decoder.
#include "testutils.h"
#include <device.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The program, lowered by lower-device: its words are the address of every
// buffer followed by its dynamic sizes, the outputs last.
extern "C" int program_job(void *hart, void *argument);

// The bench patches only this pair in a per-run ELF. All runs read one DDR
// image.
__attribute__((used, section(".bbbench.shard"),
               aligned(8))) static const volatile uint64_t shard[2] = {
    0, UINT64_MAX};

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

int main(void) {
  uint64_t bytes;
  const int64_t *lengths = (const int64_t *)file("lengths.data", &bytes);
  const uint64_t total_samples = bytes / (2 * sizeof(int64_t));
  const uint64_t first = shard[0];
  const uint64_t samples = shard[1] == UINT64_MAX ? total_samples : shard[1];
  if (first > total_samples || samples > total_samples - first)
    return 1;
  uintptr_t hartid;
  asm volatile("csrr %0, mhartid" : "=r"(hartid));
  if (hartid != DEVICE_CONTROL_HART) {
    // One program run per target token of every passage.
    for (uint64_t sample = first; sample < first + samples; ++sample)
      for (int64_t step = 0; step < lengths[2 * sample + 1]; ++step)
        program_run(program_job, NULL);
    return 0;
  }
  const uint64_t *layout = (const uint64_t *)file("cache-layout.data", &bytes);
  const uint64_t packs = layout[0], states = layout[1], total = layout[2];
  const uint64_t *strides = layout + 3;
  const uint64_t *order = strides + states;
  const char *seed = file("cache-seed.data", &bytes);
  const char *pack_names[] = {"pack-f32.data", "pack-i8.data", "pack-i32.data"};
  void **words = (void **)malloc((packs + 5 + 4 * states) * sizeof(void *));
  for (unsigned i = 0; i < packs; ++i)
    words[i] = (void *)file(pack_names[i], &bytes);
  const int64_t *contexts = (const int64_t *)file("contexts.data", &bytes);
  const uint64_t context_stride = bytes / (total_samples * sizeof(int64_t));
  const int64_t *targets = (const int64_t *)file("targets.data", &bytes);
  const uint64_t target_stride = bytes / (total_samples * sizeof(int64_t));
  const int64_t *reference = (const int64_t *)file("reference.data", &bytes);
  uint64_t capacity = 0;
  for (uint64_t sample = first; sample < first + samples; ++sample) {
    const uint64_t positions = lengths[2 * sample] + lengths[2 * sample + 1];
    if (positions > capacity)
      capacity = positions;
  }
  capacity = (capacity + 31) / 32 * 32;
  uintptr_t scratch =
      MODEL_IMAGE +
      (strtoull((const char *)MODEL_IMAGE + 9, NULL, 10) + 4095) / 4096 * 4096;
  void **cache = (void **)malloc(states * sizeof(void *));
  void **updated = (void **)malloc(states * sizeof(void *));
  for (unsigned i = 0; i < states; ++i) {
    const uint64_t size = (strides[i] * capacity + 63) / 64 * 64;
    cache[i] = (void *)scratch;
    updated[i] = (void *)(scratch + size);
    scratch += 2 * size;
  }
  float *values = (float *)scratch;
  int64_t *indices =
      (int64_t *)(scratch + (total * sizeof(float) + 63) / 64 * 64);
  uint64_t correct = 0, passed = 0, golden = 0;
  for (uint64_t sample = first; sample < first + samples; ++sample) {
    const int64_t used = lengths[2 * sample], count = lengths[2 * sample + 1];
    const int64_t *context = contexts + sample * context_stride;
    const int64_t *target = targets + sample * target_stride;
    const int64_t *expected = reference + sample * target_stride;
    const uint64_t cacheCapacity = (used + count + 31) / 32 * 32;
    // Packed KV planes can have nonzero exponent bytes in an empty cache.
    const char *pattern = seed;
    for (unsigned i = 0; i < states; ++i) {
      for (uint64_t t = 0; t < cacheCapacity; ++t)
        memcpy((char *)cache[i] + t * strides[i], pattern, strides[i]);
      pattern += strides[i];
    }
    bool right = true, same = true, checkpoint = true;
    int64_t position = 0, given = 0;
    for (int64_t step = 0; step < count; ++step) {
      // The passage first, then each target token but the last.
      const int64_t *ids = step ? &given : context;
      const uint64_t tokens = step ? 1 : used;
      unsigned word = packs;
      for (unsigned i = 0; i < 2 + states; ++i) {
        const unsigned index = order[i];
        if (index == 0) {
          words[word++] = (void *)ids;
          words[word++] = (void *)(uintptr_t)tokens;
        } else if (index == 1) {
          words[word++] = &position;
        } else {
          words[word++] = cache[index - 2];
          words[word++] = (void *)(uintptr_t)cacheCapacity;
        }
      }
      words[word++] = values;
      words[word++] = indices;
      for (unsigned i = 0; i < states; ++i) {
        words[word++] = updated[i];
        words[word++] = (void *)(uintptr_t)cacheCapacity;
      }
      if (program_run(program_job, words))
        return 1;
      for (unsigned i = 0; i < states; ++i) {
        void *swap = cache[i];
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

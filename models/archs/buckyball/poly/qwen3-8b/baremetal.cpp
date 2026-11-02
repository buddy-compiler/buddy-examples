// Generate eight tokens and compare each decoder step with the quantized
// reference.
#include "testutils.h"
#include <device.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The program, lowered by lower-device: its words are the address of every
// buffer followed by its dynamic sizes, the outputs last.
extern "C" int program_job(void *core, void *argument);

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

// The words of the program, the sizes of its inputs and outputs, their
// counts; the control hart reads them from the image.
static void *words[2 + 4 * 128];
static uint64_t sizes[2][128];
static unsigned counts[2];

int main(void) {
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart != DEVICE_CONTROL_HART) {
    for (unsigned step = 0; step < 8; ++step)
      program_run(program_job, NULL);
    return 0;
  }
  uint64_t bytes;
  const char *pack_f32 = file("pack-f32.data", &bytes);
  const char *pack_i8 = file("pack-i8.data", &bytes);
  const char *inputs = file("inputs.data", &bytes);
  const char *expected = file("outputs.data", &bytes);
  const char *text = file("sizes.txt", &bytes);
  char *outputs =
      (char *)MODEL_IMAGE +
      (strtoull((const char *)MODEL_IMAGE + 9, NULL, 10) + 4095) / 4096 * 4096;
  const char *buffers[2] = {inputs, outputs};
  unsigned inputWords[128], outputWords[128];
  unsigned word = 0;
  words[word++] = (void *)pack_f32;
  words[word++] = (void *)pack_i8;
  for (const char *line = text; line < text + bytes;
       line = strchr(line, '\n') + 1) {
    const unsigned kind = line[0] == 'o';
    char *next;
    const uint64_t size = strtoull(strchr(line, ' ') + 1, &next, 10);
    (kind ? outputWords : inputWords)[counts[kind]] = word;
    sizes[kind][counts[kind]++] = size;
    words[word++] = (void *)buffers[kind];
    while (*next == ' ')
      words[word++] = (void *)(uintptr_t)strtoull(next + 1, &next, 10);
    buffers[kind] += (size + 63) / 64 * 64;
  }
  uint64_t tokens = sizes[0][0] / sizeof(int64_t);
  const uint64_t vocabulary = sizes[1][0] / sizeof(float) / tokens;
  unsigned order[128];
  text = file("input-order.txt", &bytes);
  for (unsigned i = 0; i < counts[0]; ++i) {
    char *next;
    order[i] = strtoul(text, &next, 10);
    text = next;
  }
  int64_t position = 0, nextToken = 0;

  const uint64_t begin = read_counter();
  for (unsigned step = 0; step < 8; ++step) {
    const int status = program_run(program_job, words);
    if (status) {
      printf("program failed: %d\n", status);
      return 1;
    }
    // The values that differ from PyTorch in any bit: the logits, then the
    // keys and the values of each layer.
    uint64_t offset = 0;
    for (unsigned i = 0; i < counts[1]; ++i) {
      const uint64_t size =
          i == 0 ? tokens * vocabulary * sizeof(float) : sizes[1][i];
      for (uint64_t at = 0; at < size; at += sizeof(uint32_t)) {
        uint32_t actual, golden;
        memcpy(&actual, (const char *)words[outputWords[i]] + at,
               sizeof(actual));
        memcpy(&golden, expected + offset + at, sizeof(golden));
        if (actual != golden) {
          printf("Step %u output %u mismatch at %lu\n", step + 1, i,
                 (unsigned long)(at / sizeof(uint32_t)));
          return 1;
        }
      }
      offset += (size + 63) / 64 * 64;
    }
    expected += offset;
    const float *logits = (const float *)outputs + (tokens - 1) * vocabulary;
    uint64_t best = 0;
    for (uint64_t i = 1; i < vocabulary; ++i)
      if (logits[i] > logits[best])
        best = i;
    printf("Token %u: %lu\n", step + 1, (unsigned long)best);
    position += tokens;
    nextToken = best;
    tokens = 1;
    for (unsigned i = 0; i < counts[0]; ++i) {
      if (order[i] == 0) {
        words[inputWords[i]] = &nextToken;
        words[inputWords[i] + 1] = (void *)(uintptr_t)1;
      } else if (order[i] == 1) {
        words[inputWords[i]] = &position;
      } else {
        void *previous = words[inputWords[i]];
        words[inputWords[i]] = words[outputWords[order[i] - 1]];
        words[outputWords[order[i] - 1]] = previous;
      }
    }
    words[outputWords[0] + 1] = (void *)(uintptr_t)1;
  }
  const uint64_t elapsed = read_counter() - begin;
  printf("Generated tokens: 8\n");
  printf("Inference cycles: %lu\n", (unsigned long)elapsed);
  return 0;
}

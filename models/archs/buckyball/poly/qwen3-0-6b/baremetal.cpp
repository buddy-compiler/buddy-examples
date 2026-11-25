// Generate eight tokens greedily and compare each decoder step with the
// quantized reference. The program returns the candidates of the next token:
// the largest logits of the last position on each rank's piece of the
// vocabulary and their indices, the ranks in order; the next token is the
// first largest of them, the first maximum of the logits as torch.argmax
// picks it. The inference cycles count the program runs only.
#include "testutils.h"
#include <device.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The program, lowered by lower-device: its words are the address of every
// buffer followed by its dynamic sizes, the outputs last.
extern "C" int program_job(void *hart, void *argument);

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
static void *words[3 + 4 * 128];
static uint64_t sizes[2][128];
static unsigned counts[2];

int main(void) {
  uintptr_t hartid;
  asm volatile("csrr %0, mhartid" : "=r"(hartid));
  if (hartid != DEVICE_CONTROL_HART) {
    for (unsigned step = 0; step < 8; ++step)
      program_run(program_job, NULL);
    return 0;
  }
  uint64_t bytes;
  const char *pack_f32 = file("pack-f32.data", &bytes);
  const char *pack_i8 = file("pack-i8.data", &bytes);
  const char *pack_i32 = file("pack-i32.data", &bytes);
  const char *inputs = file("inputs.data", &bytes);
  const char *text = file("sizes.txt", &bytes);
  char *outputs =
      (char *)MODEL_IMAGE +
      (strtoull((const char *)MODEL_IMAGE + 9, NULL, 10) + 4095) / 4096 * 4096;
  const char *buffers[2] = {inputs, outputs};
  unsigned inputWords[128], outputWords[128];
  unsigned word = 0;
  words[word++] = (void *)pack_f32;
  words[word++] = (void *)pack_i8;
  words[word++] = (void *)pack_i32;
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
  unsigned order[128];
  text = file("input-order.txt", &bytes);
  for (unsigned i = 0; i < counts[0]; ++i) {
    char *next;
    order[i] = strtoul(text, &next, 10);
    text = next;
  }
  // The reference of every step (buddy-pytorch-golden.py): its candidate
  // logits and the token it picks.
  const float *goldCandidates = (const float *)file("candidates.data", &bytes);
  const int64_t *goldTokens = (const int64_t *)file("tokens.data", &bytes);
  const uint64_t candidates = sizes[1][0] / sizeof(float);
  float *stepCandidates = (float *)malloc(8 * candidates * sizeof(float));
  unsigned differentTokens = 0;
  int64_t position = 0, nextToken = 0;
  uint64_t elapsed = 0;
  for (unsigned step = 0; step < 8; ++step) {
    // CLINT time is independent of the runtime virtual mcycle.
    uint64_t begin, end;
    asm volatile("rdtime %0" : "=r"(begin)::"memory");
    const int status = program_run(program_job, words);
    asm volatile("rdtime %0" : "=r"(end)::"memory");
    elapsed += (end - begin) * DEVICE_CLINT_TICK_CYCLES;
    if (status) {
      printf("program failed: %d\n", status);
      return 1;
    }
    const float *values = (const float *)words[outputWords[0]];
    const int64_t *indices = (const int64_t *)words[outputWords[1]];
    unsigned best = 0;
    for (unsigned i = 1; i < sizes[1][0] / sizeof(float); ++i)
      if (values[i] > values[best])
        best = i;
    nextToken = indices[best];
    memcpy(stepCandidates + step * candidates, values,
           candidates * sizeof(float));
    differentTokens += nextToken != goldTokens[step];
    printf("Token %u: %ld, reference %ld\n", step + 1, (long)nextToken,
           (long)goldTokens[step]);
    position += tokens;
    tokens = 1;
    for (unsigned i = 0; i < counts[0]; ++i) {
      if (order[i] == 0) {
        words[inputWords[i]] = &nextToken;
        words[inputWords[i] + 1] = (void *)(uintptr_t)1;
      } else if (order[i] == 1) {
        words[inputWords[i]] = &position;
      } else {
        void *previous = words[inputWords[i]];
        words[inputWords[i]] = words[outputWords[order[i]]];
        words[outputWords[order[i]]] = previous;
      }
    }
  }
  printf("Generated tokens: 8, %u differ from the reference\n",
         differentTokens);
  printf("Inference cycles: %lu\n", (unsigned long)elapsed);
  // The reference runs the device program's quantized arithmetic in
  // PyTorch: the device differs in the FP32 order of its sums and in the
  // rounding of the requantizations that follow them.
  const int failed =
      check_output(stepCandidates, goldCandidates, 8 * candidates, 1e-2f);
  return failed || differentTokens;
}

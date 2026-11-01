// Qwen3-0.6B on the poly device, baremetal: every hart enters main; the
// control hart runs the device program, one decoder step of a short prompt
// written from position 0 into an empty KV cache, which launches the
// semantic graphs on the reduce core and the ranks of the poly cores, and
// checks the logits and the updated KV cache against the PyTorch program of
// the same quantized and partitioned graphs.
//
// The parameters, the inputs and the expected outputs are the files of the
// model image at the model region of the DDR (MODEL_IMAGE), packed by
// model-pack: pack-f32.data, pack-i8.data, inputs.data and outputs.data (the
// values one after the other at 64-byte boundaries, in the order of the
// program), sizes.txt (the bytes of each, `input <bytes>` and
// `output <bytes>` lines). The outputs are written after the image.
#include <device.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define X(hart) +1
static const unsigned core_count = 0 DEVICE_CORES;
#undef X

// The program, lowered by lower-device: its words are the address of every
// buffer followed by its dynamic sizes, the outputs last.
int program_job(void *core, void *argument);

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
static void *words[256];
static uint64_t sizes[2][128];
static unsigned counts[2];

int main(void) {
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart != DEVICE_CONTROL_HART) {
    program_run(program_job, NULL);
    return 0;
  }
  uint64_t bytes;
  const char *pack_f32 = file("pack-f32.data", &bytes);
  const char *pack_i8 = file("pack-i8.data", &bytes);
  const char *inputs = file("inputs.data", &bytes);
  const char *expected = file("outputs.data", &bytes);
  const char *text = file("sizes.txt", &bytes);
  for (const char *line = text; line < text + bytes;
       line = strchr(line, '\n') + 1) {
    const unsigned kind = line[0] == 'o';
    sizes[kind][counts[kind]++] = strtoull(strchr(line, ' ') + 1, NULL, 10);
  }
  // The outputs after the image, whose size is on its first line.
  char *outputs =
      (char *)MODEL_IMAGE +
      (strtoull((const char *)MODEL_IMAGE + 9, NULL, 10) + 4095) / 4096 * 4096;

  // The tokens [1, count] and the logits [1, count, vocabulary] have the
  // token count as their dynamic size.
  const uint64_t tokens = sizes[0][0] / 8;
  unsigned word = 0;
  words[word++] = (void *)pack_f32;
  words[word++] = (void *)pack_i8;
  const char *input = inputs;
  for (unsigned i = 0; i < counts[0]; ++i) {
    words[word++] = (void *)input;
    if (i == 0)
      words[word++] = (void *)(uintptr_t)tokens;
    input += (sizes[0][i] + 63) / 64 * 64;
  }
  char *output = outputs;
  for (unsigned i = 0; i < counts[1]; ++i) {
    words[word++] = output;
    if (i == 0)
      words[word++] = (void *)(uintptr_t)tokens;
    output += (sizes[1][i] + 63) / 64 * 64;
  }

  const int status = program_run(program_job, words);
  if (status) {
    printf("program failed: %d\n", status);
    return 1;
  }
  printf("%lu tokens\n", (unsigned long)tokens);
  // The values that differ from PyTorch in any bit: the logits, then the
  // keys and the values of each layer.
  unsigned different = 0;
  uint64_t offset = 0;
  for (unsigned i = 0; i < counts[1]; ++i) {
    unsigned count = 0;
    for (uint64_t at = 0; at < sizes[1][i]; at += 4)
      count += memcmp(outputs + offset + at, expected + offset + at, 4) != 0;
    printf("output %u: %u of %lu differ from PyTorch\n", i, count,
           (unsigned long)(sizes[1][i] / 4));
    different += count;
    offset += (sizes[1][i] + 63) / 64 * 64;
  }
  printf("latency: %lu cycles\n", (unsigned long)program_latency());
  for (unsigned core = 0; core < core_count; ++core)
    printf("core %u: jobs took %lu cycles\n", core,
           (unsigned long)program_busy(core));
  return different != 0;
}

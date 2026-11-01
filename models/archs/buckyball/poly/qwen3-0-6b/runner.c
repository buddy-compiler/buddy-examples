// The tiny Qwen3 on the poly device, baremetal: every hart enters main; the
// control hart runs the device program, which launches the segments of the
// reduce core and of the 8 ranks on the cores, and checks the logits and the
// updated KV cache against the PyTorch program of the same quantized and
// partitioned graphs. The program takes any number of tokens: it runs twice,
// with 4 tokens and with 9 (output/tiny/<count>/: the inputs,
// output<i>.data).
#include <device.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const unsigned char pack_f32[] __attribute__((aligned(64))) = {
#embed "pack-f32.data"
};
static const unsigned char pack_i8[] __attribute__((aligned(64))) = {
#embed "pack-i8.data"
};
// The inputs of each run in the order of the program: tokens, start, then
// the keys and the values of each layer; its outputs: the logits, then the
// keys and the values of each layer.
static const unsigned char input4_0[] __attribute__((aligned(64))) = {
#embed "4/input0.data"
};
static const unsigned char input4_1[] __attribute__((aligned(64))) = {
#embed "4/input1.data"
};
static const unsigned char input4_2[] __attribute__((aligned(64))) = {
#embed "4/input2.data"
};
static const unsigned char input4_3[] __attribute__((aligned(64))) = {
#embed "4/input3.data"
};
static const unsigned char input4_4[] __attribute__((aligned(64))) = {
#embed "4/input4.data"
};
static const unsigned char input4_5[] __attribute__((aligned(64))) = {
#embed "4/input5.data"
};
static const unsigned char expected4_0[] __attribute__((aligned(64))) = {
#embed "4/output0.data"
};
static const unsigned char expected4_1[] __attribute__((aligned(64))) = {
#embed "4/output1.data"
};
static const unsigned char expected4_2[] __attribute__((aligned(64))) = {
#embed "4/output2.data"
};
static const unsigned char expected4_3[] __attribute__((aligned(64))) = {
#embed "4/output3.data"
};
static const unsigned char expected4_4[] __attribute__((aligned(64))) = {
#embed "4/output4.data"
};
static const unsigned char input9_0[] __attribute__((aligned(64))) = {
#embed "9/input0.data"
};
static const unsigned char input9_1[] __attribute__((aligned(64))) = {
#embed "9/input1.data"
};
static const unsigned char input9_2[] __attribute__((aligned(64))) = {
#embed "9/input2.data"
};
static const unsigned char input9_3[] __attribute__((aligned(64))) = {
#embed "9/input3.data"
};
static const unsigned char input9_4[] __attribute__((aligned(64))) = {
#embed "9/input4.data"
};
static const unsigned char input9_5[] __attribute__((aligned(64))) = {
#embed "9/input5.data"
};
static const unsigned char expected9_0[] __attribute__((aligned(64))) = {
#embed "9/output0.data"
};
static const unsigned char expected9_1[] __attribute__((aligned(64))) = {
#embed "9/output1.data"
};
static const unsigned char expected9_2[] __attribute__((aligned(64))) = {
#embed "9/output2.data"
};
static const unsigned char expected9_3[] __attribute__((aligned(64))) = {
#embed "9/output3.data"
};
static const unsigned char expected9_4[] __attribute__((aligned(64))) = {
#embed "9/output4.data"
};
// The outputs of a run, as large as those of the longer one.
static float output0[sizeof(expected9_0) / 4] __attribute__((aligned(64)));
static float output1[sizeof(expected9_1) / 4] __attribute__((aligned(64)));
static float output2[sizeof(expected9_2) / 4] __attribute__((aligned(64)));
static float output3[sizeof(expected9_3) / 4] __attribute__((aligned(64)));
static float output4[sizeof(expected9_4) / 4] __attribute__((aligned(64)));

#define X(hart) +1
static const unsigned core_count = 0 DEVICE_CORES;
#undef X

// The program, lowered by lower-device: its words are the address of
// every buffer followed by its dynamic sizes, the outputs last.
int program_job(void *core, void *argument);

// The values of `got` that differ from `want` in any bit.
static unsigned compare(const char *name, const float *got,
                        const unsigned char *want, unsigned count) {
  unsigned different = 0;
  for (unsigned i = 0; i < count; ++i)
    different += memcmp(&got[i], want + 4 * i, 4) != 0;
  printf("%s: %u of %u differ from PyTorch\n", name, different, count);
  return different;
}

int main(void) {
  // The tokens [1, count] and the logits [1, count, 512] have the token
  // count as their dynamic size.
  static void *run4[] = {(void *)pack_f32,
                         (void *)pack_i8,
                         (void *)input4_0,
                         (void *)(uintptr_t)4,
                         (void *)input4_1,
                         (void *)input4_2,
                         (void *)input4_3,
                         (void *)input4_4,
                         (void *)input4_5,
                         output0,
                         (void *)(uintptr_t)4,
                         output1,
                         output2,
                         output3,
                         output4};
  static void *run9[] = {(void *)pack_f32,
                         (void *)pack_i8,
                         (void *)input9_0,
                         (void *)(uintptr_t)9,
                         (void *)input9_1,
                         (void *)input9_2,
                         (void *)input9_3,
                         (void *)input9_4,
                         (void *)input9_5,
                         output0,
                         (void *)(uintptr_t)9,
                         output1,
                         output2,
                         output3,
                         output4};
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  unsigned different = 0;

  const int status4 = program_run(program_job, run4);
  if (hart == DEVICE_CONTROL_HART) {
    if (status4) {
      printf("program failed: %d\n", status4);
      return 1;
    }
    printf("4 tokens\n");
    different +=
        compare("logits", output0, expected4_0, sizeof(expected4_0) / 4);
    different +=
        compare("keys 0", output1, expected4_1, sizeof(expected4_1) / 4);
    different +=
        compare("keys 1", output2, expected4_2, sizeof(expected4_2) / 4);
    different +=
        compare("values 0", output3, expected4_3, sizeof(expected4_3) / 4);
    different +=
        compare("values 1", output4, expected4_4, sizeof(expected4_4) / 4);
    printf("latency: %lu cycles\n", (unsigned long)program_latency());
  }

  const int status9 = program_run(program_job, run9);
  if (hart != DEVICE_CONTROL_HART)
    return 0;
  if (status9) {
    printf("program failed: %d\n", status9);
    return 1;
  }
  printf("9 tokens\n");
  different += compare("logits", output0, expected9_0, sizeof(expected9_0) / 4);
  different += compare("keys 0", output1, expected9_1, sizeof(expected9_1) / 4);
  different += compare("keys 1", output2, expected9_2, sizeof(expected9_2) / 4);
  different +=
      compare("values 0", output3, expected9_3, sizeof(expected9_3) / 4);
  different +=
      compare("values 1", output4, expected9_4, sizeof(expected9_4) / 4);
  printf("latency: %lu cycles\n", (unsigned long)program_latency());
  for (unsigned core = 0; core < core_count; ++core)
    printf("core %u: jobs took %lu cycles\n", core,
           (unsigned long)program_busy(core));
  return different != 0;
}

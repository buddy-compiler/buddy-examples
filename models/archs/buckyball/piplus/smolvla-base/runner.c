// smolvla-base on the piplus device, baremetal: every hart enters main; the
// control hart runs the program, which launches the graphs on the dense and
// action cores, and checks its actions on one SO100 observation against the
// PyTorch program of the same quantized graphs (output/actions.data).
#include <device.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const unsigned char vision_f32[] __attribute__((aligned(64))) = {
#embed "vision-arg0.data"
};
static const unsigned char vision_i8[] __attribute__((aligned(64))) = {
#embed "vision-arg1.data"
};
static const unsigned char prefix_f32[] __attribute__((aligned(64))) = {
#embed "prefix-arg0.data"
};
static const unsigned char prefix_i8[] __attribute__((aligned(64))) = {
#embed "prefix-arg1.data"
};
static const unsigned char step_f32[] __attribute__((aligned(64))) = {
#embed "step-arg0.data"
};
static const unsigned char step_i8[] __attribute__((aligned(64))) = {
#embed "step-arg1.data"
};
static const unsigned char images[] __attribute__((aligned(64))) = {
#embed "images.data"
};
static const unsigned char lang_tokens[] __attribute__((aligned(64))) = {
#embed "lang_tokens.data"
};
static const unsigned char lang_masks[] __attribute__((aligned(64))) = {
#embed "lang_masks.data"
};
static const unsigned char state[] __attribute__((aligned(64))) = {
#embed "state.data"
};
static const unsigned char noise[] __attribute__((aligned(64))) = {
#embed "noise.data"
};
static const unsigned char reference[] __attribute__((aligned(64))) = {
#embed "actions.data"
};
static float actions[sizeof(reference) / sizeof(float)]
    __attribute__((aligned(64)));

#define X(hart) +1
static const unsigned core_count = 0 DEVICE_CORES;
#undef X

// The program, lowered by lower-device: its arguments are the
// pointers of its buffers, the actions last.
int program_job(void *core, void *argument);

int main(void) {
  static void *arguments[] = {
      (void *)vision_f32, (void *)vision_i8,   (void *)prefix_f32,
      (void *)prefix_i8,  (void *)step_f32,    (void *)step_i8,
      (void *)images,     (void *)lang_tokens, (void *)lang_masks,
      (void *)state,      (void *)noise,       actions};
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  const int status = program_run(program_job, arguments);
  if (hart != DEVICE_CONTROL_HART)
    return 0;
  if (status) {
    printf("program failed: %d\n", status);
    return 1;
  }
  const float *expected = (const float *)reference;
  unsigned different = 0;
  float largest = 0;
  for (unsigned i = 0; i < sizeof(actions) / sizeof(float); ++i) {
    uint32_t got, want;
    memcpy(&got, &actions[i], 4);
    memcpy(&want, &expected[i], 4);
    different += got != want;
    float difference = actions[i] - expected[i];
    if (difference < 0)
      difference = -difference;
    if (difference > largest)
      largest = difference;
  }
  printf("actions: %u of %u differ from PyTorch, max |difference| %d.%09d\n",
         different, (unsigned)(sizeof(actions) / sizeof(float)), (int)largest,
         (int)((largest - (int)largest) * 1e9f));
  printf("latency: %lu cycles\n", (unsigned long)program_latency());
  for (unsigned core = 0; core < core_count; ++core)
    printf("core %u: jobs took %lu cycles\n", core,
           (unsigned long)program_busy(core));
  return 0;
}

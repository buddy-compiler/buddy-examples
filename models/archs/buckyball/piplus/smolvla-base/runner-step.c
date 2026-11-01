// The first denoising step of smolvla-base alone on the piplus device,
// baremetal: every hart enters main; the control hart runs the step program,
// which launches the step graph on an action core with the inputs the
// quantized program gives it on one SO100 observation (the noise, time 1,
// the prefix padding mask and its keys and values), and checks the velocity
// against the PyTorch step of the same quantized graph
// (output/step_velocity.data).
#include <device.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const unsigned char step_f32[] __attribute__((aligned(64))) = {
#embed "step-arg0.data"
};
static const unsigned char step_i8[] __attribute__((aligned(64))) = {
#embed "step-arg1.data"
};
static const unsigned char noise[] __attribute__((aligned(64))) = {
#embed "noise.data"
};
static const unsigned char time_[] __attribute__((aligned(64))) = {
#embed "step_time.data"
};
static const unsigned char pad_masks[] __attribute__((aligned(64))) = {
#embed "step_pad_masks.data"
};
// The keys of every layer, then their values: 32 tensors of 1x241x5x64.
static const unsigned char cache[] __attribute__((aligned(64))) = {
#embed "step_cache.data"
};
static const unsigned char reference[] __attribute__((aligned(64))) = {
#embed "step_velocity.data"
};
static float velocity[sizeof(reference) / sizeof(float)]
    __attribute__((aligned(64)));

#define CACHE_TENSORS 32

#define X(hart) +1
static const unsigned core_count = 0 DEVICE_CORES;
#undef X

// The step program, lowered by lower-device: its arguments are the
// pointers of its buffers, the velocity last.
int step_program_job(void *core, void *argument);

int main(void) {
  static void *arguments[5 + CACHE_TENSORS + 1];
  arguments[0] = (void *)step_f32;
  arguments[1] = (void *)step_i8;
  arguments[2] = (void *)noise;
  arguments[3] = (void *)time_;
  arguments[4] = (void *)pad_masks;
  for (unsigned i = 0; i < CACHE_TENSORS; ++i)
    arguments[5 + i] = (void *)(cache + i * (sizeof(cache) / CACHE_TENSORS));
  arguments[5 + CACHE_TENSORS] = velocity;
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  const int status = program_run(step_program_job, arguments);
  if (hart != DEVICE_CONTROL_HART)
    return 0;
  if (status) {
    printf("step program failed: %d\n", status);
    return 1;
  }
  const float *expected = (const float *)reference;
  unsigned different = 0;
  float largest = 0;
  for (unsigned i = 0; i < sizeof(velocity) / sizeof(float); ++i) {
    uint32_t got, want;
    memcpy(&got, &velocity[i], 4);
    memcpy(&want, &expected[i], 4);
    different += got != want;
    float difference = velocity[i] - expected[i];
    if (difference < 0)
      difference = -difference;
    if (difference > largest)
      largest = difference;
  }
  printf("velocity: %u of %u differ from PyTorch, max |difference| %d.%09d\n",
         different, (unsigned)(sizeof(velocity) / sizeof(float)), (int)largest,
         (int)((largest - (int)largest) * 1e9f));
  printf("latency: %lu cycles\n", (unsigned long)program_latency());
  for (unsigned core = 0; core < core_count; ++core)
    printf("core %u: jobs took %lu cycles\n", core,
           (unsigned long)program_busy(core));
  return 0;
}

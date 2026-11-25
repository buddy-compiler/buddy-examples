// The first denoising step of smolvla-base alone on the piplus device,
// baremetal: every hart enters main; the control hart runs the step program,
// which launches the step's graphs, partitioned across the action and dense
// harts (frontend/partition), with the inputs the
// quantized program gives it on one SO100 observation (the noise, time 1,
// the prefix's masks and angles for the steps and its keys and values), and
// checks the velocity against the PyTorch step of the same quantized graph
// (output/step_velocity.data).
#include "testutils.h"
#include <device.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const unsigned char step_f32[] __attribute__((aligned(64))) = {
#embed "step-pack-f32.data"
};
static const unsigned char step_i8[] __attribute__((aligned(64))) = {
#embed "step-pack-i8.data"
};
static const unsigned char noise[] __attribute__((aligned(64))) = {
#embed "noise.data"
};
static const unsigned char time_[] __attribute__((aligned(64))) = {
#embed "step_time.data"
};
static const unsigned char masks[] __attribute__((aligned(64))) = {
#embed "step_masks.data"
};
static const unsigned char angles[] __attribute__((aligned(64))) = {
#embed "step_angles.data"
};
// The keys of every layer, then their values: 32 tensors of 1x256x5x64.
static const unsigned char cache[] __attribute__((aligned(64))) = {
#embed "step_cache.data"
};
static const unsigned char reference[] __attribute__((aligned(64))) = {
#embed "step_velocity.data"
};
static float velocity[sizeof(reference) / sizeof(float)]
    __attribute__((aligned(64)));

#define CACHE_TENSORS 32

// The step program, lowered by lower-device: its arguments are the
// pointers of its buffers, the velocity last.
extern "C" int step_program_job(void *hart, void *argument);

int main(void) {
  uintptr_t hartid;
  asm volatile("csrr %0, mhartid" : "=r"(hartid));
  static void *arguments[6 + CACHE_TENSORS + 1];
  if (hartid == DEVICE_CONTROL_HART) {
    arguments[0] = (void *)step_f32;
    arguments[1] = (void *)step_i8;
    arguments[2] = (void *)noise;
    arguments[3] = (void *)time_;
    arguments[4] = (void *)masks;
    arguments[5] = (void *)angles;
    for (unsigned i = 0; i < CACHE_TENSORS; ++i)
      arguments[6 + i] = (void *)(cache + i * (sizeof(cache) / CACHE_TENSORS));
    arguments[6 + CACHE_TENSORS] = velocity;
  }
  const uint64_t begin = read_counter();
  const int status = program_run(step_program_job, arguments);
  const uint64_t elapsed = read_counter() - begin;
  if (hartid != DEVICE_CONTROL_HART)
    return 0;
  if (status) {
    printf("step program failed: %d\n", status);
    return 1;
  }
  // The velocity against PyTorch (reference), bit by bit and at most.
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
  printf("velocity: %u of %u differ from PyTorch, max |difference| "
         "%d.%09d\n",
         different, (unsigned)(sizeof(velocity) / sizeof(float)), (int)largest,
         (int)((largest - (int)largest) * 1e9f));
  printf("Inference cycles: %lu\n", (unsigned long)elapsed);
  return 0;
}

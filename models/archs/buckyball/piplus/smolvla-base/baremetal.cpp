#include "ModelImage.h"
#include "model.h"
#include "testutils.h"
#include <device.h>

extern "C" int program_job(void *, void *);
static float actions[ACTION_ELEMENTS] __attribute__((aligned(64)));

int main() {
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  void *arguments[12];
  const char *files[] = {"ddr/vision-arg0.data", "ddr/vision-arg1.data",
                         "ddr/prefix-arg0.data", "ddr/prefix-arg1.data",
                         "ddr/step-arg0.data",   "ddr/step-arg1.data"};
  uint64_t bytes;
  if (hart == DEVICE_CONTROL_HART) {
    for (unsigned i = 0; i < 6; ++i)
      arguments[i] =
          const_cast<char *>(model::imageFile(MODEL_IMAGE, files[i], &bytes));
    const auto &input = *reinterpret_cast<const Observation *>(
        model::imageFile(MODEL_IMAGE, "observation.data", &bytes));
    arguments[6] = const_cast<float *>(input.images);
    arguments[7] = const_cast<int64_t *>(input.tokens);
    arguments[8] = const_cast<int64_t *>(input.masks);
    arguments[9] = const_cast<float *>(input.state);
    arguments[10] = const_cast<float *>(input.noise);
    arguments[11] = actions;
  }
  const uint64_t begin = read_counter();
  const int status = program_run(program_job, arguments);
  const uint64_t elapsed = read_counter() - begin;
  if (hart != DEVICE_CONTROL_HART)
    return 0;
  if (status)
    return 1;
  const auto *expected = reinterpret_cast<const float *>(
      model::imageFile(MODEL_IMAGE, "actions.data", &bytes));
  unsigned different = 0;
  for (unsigned i = 0; i < ACTION_ELEMENTS; ++i)
    different += memcmp(actions + i, expected + i, sizeof(float)) != 0;
  printf("Actions: %u\nInference cycles: %lu\n", ACTION_ELEMENTS,
         static_cast<unsigned long>(elapsed));
  if (different)
    printf("Reference mismatch: %u action values\n", different);
  return different != 0;
}

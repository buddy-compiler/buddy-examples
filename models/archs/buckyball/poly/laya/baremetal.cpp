#include "ModelImage.h"
#include "model.h"
#include "testutils.h"
#include <program.h>

extern "C" int program_job(void *, void *);
static float choices[16], actions[2];

int main() {
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart != DEVICE_CONTROL_HART)
    return program_run(program_job, nullptr);
  uint64_t bytes;
  auto *parameters = const_cast<char *>(
      model::imageFile(MODEL_IMAGE, "ddr/pack-f32.data", &bytes));
  auto *weights = const_cast<char *>(
      model::imageFile(MODEL_IMAGE, "ddr/pack-i8.data", &bytes));
  const auto &input = *reinterpret_cast<const Input *>(
      model::imageFile(MODEL_IMAGE, "input.data", &bytes));
  void *arguments[] = {parameters,
                       weights,
                       const_cast<int64_t *>(input.tokens),
                       (void *)128,
                       const_cast<int64_t *>(input.mask),
                       (void *)128,
                       const_cast<int64_t *>(input.positions),
                       const_cast<bool *>(input.valid),
                       const_cast<int64_t *>(input.type),
                       choices,
                       actions};
  const uint64_t begin = read_counter();
  const int status = program_run(program_job, arguments);
  const uint64_t elapsed = read_counter() - begin;
  if (status)
    return 1;
  const auto *expected = reinterpret_cast<const float *>(
      model::imageFile(MODEL_IMAGE, "expected.data", &bytes));
  unsigned choice = 0, gold = 0;
  for (unsigned i = 1; i < 16; ++i) {
    if (choices[i] > choices[choice])
      choice = i;
    if (expected[i] > expected[gold])
      gold = i;
  }
  const unsigned action = actions[1] > actions[0];
  printf("Choice: %u\nAction: %u\nInference cycles: %lu\n", choice, action,
         static_cast<unsigned long>(elapsed));
  return choice != gold || action != (expected[17] > expected[16]);
}

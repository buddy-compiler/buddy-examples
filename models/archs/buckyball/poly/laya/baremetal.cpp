#include "ModelImage.h"
#include "model.h"
#include "testutils.h"
#include <program.h>

extern "C" int program_job(void *, void *);
// The 16 choice scores, then the 2 action scores, as expected.data.
alignas(64) static float scores[18];
static float *const choices = scores, *const actions = scores + 16;

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
  auto *table = const_cast<char *>(
      model::imageFile(MODEL_IMAGE, "ddr/pack-i32.data", &bytes));
  const auto &input = *reinterpret_cast<const Input *>(
      model::imageFile(MODEL_IMAGE, "input.data", &bytes));
  void *arguments[] = {parameters,
                       weights,
                       table,
                       const_cast<int64_t *>(input.tokens),
                       (void *)128,
                       const_cast<int64_t *>(input.mask),
                       (void *)128,
                       const_cast<int64_t *>(input.positions),
                       const_cast<bool *>(input.valid),
                       const_cast<int64_t *>(input.type),
                       choices,
                       actions};
  // CLINT time is independent of the runtime virtual mcycle.
  uint64_t begin, end;
  asm volatile("rdtime %0" : "=r"(begin)::"memory");
  const int status = program_run(program_job, arguments);
  asm volatile("rdtime %0" : "=r"(end)::"memory");
  const uint64_t elapsed = (end - begin) * DEVICE_CLINT_TICK_CYCLES;
  if (status)
    return 1;
  // The scores of the quantized model in PyTorch (expected.data): the 16
  // choices, then the 2 actions.
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
  const unsigned goldAction = expected[17] > expected[16];
  printf("Choice: %u, reference %u\nAction: %u, reference %u\n"
         "Inference cycles: %lu\n",
         choice, gold, action, goldAction, static_cast<unsigned long>(elapsed));
  // The valid choices and the actions; an invalid choice is the mask
  // constant on both sides. The reference runs the same quantized
  // arithmetic: the device differs in the FP32 order of its sums and in
  // the rounding of the requantizations that follow them.
  float actual[18], golden[18];
  unsigned count = 0;
  for (unsigned i = 0; i < 18; ++i)
    if (i >= 16 || input.valid[i]) {
      actual[count] = scores[i];
      golden[count++] = expected[i];
    }
  const int failed = check_output(actual, golden, count, 1e-2f);
  return failed || choice != gold || action != goldAction;
}

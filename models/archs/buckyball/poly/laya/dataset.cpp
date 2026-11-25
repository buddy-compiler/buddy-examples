// The accuracy run of Laya on poly: the held-out typed decisions of the model
// region (a shard of them; bench-device --bemu-accuracy packs it beside the
// packs into the DDR image of each run), each as baremetal.cpp runs its one.
// Prints one line: the samples, those choosing the labelled option (task
// accuracy), those whose choice and action are the reference's (the pass
// rate) and those the reference chooses as labelled (the accuracy of PyTorch
// on the same samples).
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
  // The samples of this run, in the model region.
  const auto *inputs = reinterpret_cast<const Input *>(
      model::imageFile(MODEL_IMAGE, "inputs.data", &bytes));
  const auto *references = reinterpret_cast<const float *>(
      model::imageFile(MODEL_IMAGE, "reference.data", &bytes));
  const auto *labels = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_IMAGE, "labels.data", &bytes));
  const uint64_t samples = bytes / sizeof(int64_t);
  uint64_t correct = 0, passed = 0, golden = 0;
  for (uint64_t sample = 0; sample < samples; ++sample) {
    const Input &input = inputs[sample];
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
    if (program_run(program_job, arguments))
      return 1;
    const float *expected = references + sample * 18;
    unsigned choice = 0, gold = 0;
    for (unsigned i = 1; i < 16; ++i) {
      if (choices[i] > choices[choice])
        choice = i;
      if (expected[i] > expected[gold])
        gold = i;
    }
    const unsigned action = actions[1] > actions[0];
    const unsigned goldAction = expected[17] > expected[16];
    correct += choice == labels[sample];
    golden += gold == labels[sample];
    // A sample passes when its choice and action are the reference's: the
    // scores' rounding noise grows through the online conversions.
    passed += choice == gold && action == goldAction;
  }
  printf(
      "Dataset: %lu samples, %lu correct, %lu passed, %lu reference "
      "correct\n",
      static_cast<unsigned long>(samples), static_cast<unsigned long>(correct),
      static_cast<unsigned long>(passed), static_cast<unsigned long>(golden));
  return 0;
}

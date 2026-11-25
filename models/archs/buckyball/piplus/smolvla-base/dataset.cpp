// The accuracy run of SmolVLA: the SO100 frames of the model region (a shard
// of them; bench-device --bemu-accuracy packs it beside the parameters into
// the DDR image of each run), each as baremetal.cpp runs its one. Prints two
// lines: the samples and those whose actions are the reference's within the
// tolerance of baremetal.cpp (the pass rate), with no labelled class
// ("0 correct"); and the squared errors of the device's and the reference's
// actions against the recorded ones over the 6 joints of the robot.
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
  const char *files[] = {"ddr/vision-arg0.data",     "ddr/vision-arg1.data",
                         "ddr/prefix-pack-f32.data", "ddr/prefix-pack-i8.data",
                         "ddr/step-pack-f32.data",   "ddr/step-pack-i8.data"};
  uint64_t bytes;
  for (unsigned i = 0; i < 6; ++i)
    arguments[i] =
        const_cast<char *>(model::imageFile(MODEL_IMAGE, files[i], &bytes));
  const auto *inputs = reinterpret_cast<const Observation *>(
      model::imageFile(MODEL_IMAGE, "observations.data", &bytes));
  const auto *references = reinterpret_cast<const float *>(
      model::imageFile(MODEL_IMAGE, "reference.data", &bytes));
  const auto *truths = reinterpret_cast<const float *>(
      model::imageFile(MODEL_IMAGE, "truth.data", &bytes));
  model::imageFile(MODEL_IMAGE, "labels.data", &bytes);
  const uint64_t samples = bytes / sizeof(int64_t);
  // Every hart runs the program of every sample.
  uint64_t passed = 0;
  double device = 0, reference = 0;
  const unsigned steps = ACTION_ELEMENTS / ACTION_DIM;
  for (uint64_t sample = 0; sample < samples; ++sample) {
    const Observation &input = inputs[sample];
    arguments[6] = const_cast<float *>(input.images);
    arguments[7] = const_cast<int64_t *>(input.tokens);
    arguments[8] = const_cast<int64_t *>(input.masks);
    arguments[9] = const_cast<float *>(input.state);
    arguments[10] = const_cast<float *>(input.noise);
    arguments[11] = actions;
    if (program_run(program_job, arguments))
      return 1;
    if (hart != DEVICE_CONTROL_HART)
      continue;
    const float *expected = references + sample * ACTION_ELEMENTS;
    passed += !check_output(actions, expected, ACTION_ELEMENTS, 0.06f);
    const float *truth = truths + sample * steps * 6;
    for (unsigned k = 0; k < steps; ++k)
      for (unsigned d = 0; d < 6; ++d) {
        const double a = actions[k * ACTION_DIM + d] - truth[k * 6 + d];
        const double r = expected[k * ACTION_DIM + d] - truth[k * 6 + d];
        device += a * a;
        reference += r * r;
      }
  }
  if (hart != DEVICE_CONTROL_HART)
    return 0;
  printf("Dataset: %lu samples, 0 correct, %lu passed, 0 reference correct\n",
         static_cast<unsigned long>(samples),
         static_cast<unsigned long>(passed));
  printf("Squared error: ");
  print_decimal(static_cast<float>(device));
  printf(" ");
  print_decimal(static_cast<float>(reference));
  printf(" %lu\n", static_cast<unsigned long>(samples * steps * 6));
  return 0;
}

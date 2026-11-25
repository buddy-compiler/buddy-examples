#include "ModelImage.h"
#include "model.h"
#include "testutils.h"
#include <cstring>
#include <device.h>

extern "C" int program_job(void *, void *);
static float actions[ACTION_ELEMENTS] __attribute__((aligned(64)));

// The final output on the SCU's output stream of the control hart, for the
// checks of the simulations: the first run captures it, later runs compare it.
static const uint8_t reference[sizeof(actions)]
    __attribute__((section(".bbsim.reference"), used)) = {};
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} capture __attribute__((section(".bbsim.capture_output"), used, retain)) = {
    nullptr, reference, sizeof(reference)};

static void stream_output(uintptr_t hart) {
  auto stream =
      reinterpret_cast<volatile uint32_t *>(0x60000000ULL + hart * 0x40000);
  *stream = 0x42424f55;
  for (unsigned i = 0; i < sizeof(actions) / 4; ++i) {
    uint32_t word;
    memcpy(&word, reinterpret_cast<const uint8_t *>(actions) + 4 * i,
           sizeof(word));
    *stream = word;
  }
}

int main() {
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  void *arguments[12];
  const char *files[] = {"ddr/vision-arg0.data",     "ddr/vision-arg1.data",
                         "ddr/prefix-pack-f32.data", "ddr/prefix-pack-i8.data",
                         "ddr/step-pack-f32.data",   "ddr/step-pack-i8.data"};
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
  printf("Actions: %u\nInference cycles: %lu\n", ACTION_ELEMENTS,
         static_cast<unsigned long>(elapsed));
  // Where the cycles went: the program on the virtual clocks of the harts
  // and each hart in its own graphs.
  printf("Program latency: %lu\n",
         static_cast<unsigned long>(program_latency()));
#define X(index)                                                               \
  printf("Hart %u busy: %lu\n", static_cast<unsigned>(index),                  \
         static_cast<unsigned long>(program_busy(index)));
  DEVICE_HARTS
#undef X
  // The actions against the FP32 program of the graphs in PyTorch
  // (actions.data): the quantized program in PyTorch takes over an hour. The
  // tolerance is the quantization error buddy-pytorch-calibrate.py accepts
  // against FP32 (0.06).
  const auto *expected = reinterpret_cast<const float *>(
      model::imageFile(MODEL_IMAGE, "actions.data", &bytes));
  const int failed = check_output(actions, expected, ACTION_ELEMENTS, 0.06f);
  stream_output(hart);
  return failed;
}

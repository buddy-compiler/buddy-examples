// A GVN slice with static input and host-side output verification.
// BEMU captures the output; the generated ELF checks that reference on RTL.
#include "MemRef.h"

#include <stdint.h>

using model::MemRef;

extern "C" void _mlir_ciface_forward(MemRef<float, 2> *,
                                     MemRef<const float, 1> *,
                                     MemRef<const int32_t, 1> *,
                                     MemRef<const int8_t, 1> *,
                                     MemRef<const float, 4> *);

extern "C" {
alignas(64) unsigned char gvn_in[] = {
#embed "gvn_in.bin"
};
alignas(64) unsigned char gvn_table[] = {
#embed "gvn_table.bin"
};
// The expected values: an iteration writes only its part of the output.
alignas(64) unsigned char gvn_out[] = {
#embed "gvn_expect.bin"
};
}

alignas(64) static const unsigned char expected[] = {
#embed "gvn_expect.bin"
};

alignas(64) static const unsigned char parameters[] = {
#embed "arg0.data"
};
alignas(64) static const unsigned char biases[] = {
#embed "arg1.data"
};
alignas(64) static const unsigned char weights[] = {
#embed "arg2.data"
};
alignas(64) static const unsigned char pixels[] = {
#embed "inputs.data"
};
static_assert(sizeof(pixels) == 224 * 224 * 3 * sizeof(float));

// BEMU captures the reference; the FPGA runner checks the streamed output.
__attribute__((used, section(".bbsim.capture_output"),
               aligned(8))) static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check = {gvn_out, expected, sizeof(expected)};
extern "C" void _mlir_ciface_buddyTraceCycleStartPath(int64_t, int64_t, int64_t,
                                                      int64_t, int64_t,
                                                      int64_t);
extern "C" void _mlir_ciface_buddyTraceCycleEndPath(int64_t, int64_t, int64_t,
                                                    int64_t, int64_t, int64_t);

int main() {
  const auto *p = reinterpret_cast<const float *>(parameters);
  const auto *b = reinterpret_cast<const int32_t *>(biases);
  const auto *w = reinterpret_cast<const int8_t *>(weights);
  const auto *image = reinterpret_cast<const float *>(pixels);
  MemRef<const float, 1> f32{p, p, 0, {sizeof(parameters) / 4}, {1}};
  MemRef<const int32_t, 1> i32{b, b, 0, {sizeof(biases) / 4}, {1}};
  MemRef<const int8_t, 1> i8{w, w, 0, {sizeof(weights)}, {1}};
  MemRef<const float, 4> input{
      image, image, 0, {1, 224, 224, 3}, {150528, 672, 3, 1}};
  MemRef<float, 2> output{};
  _mlir_ciface_buddyTraceCycleStartPath(INT64_MAX, 1, INT64_MAX, 0, 0, 0);
  _mlir_ciface_forward(&output, &f32, &i32, &i8, &input);
  _mlir_ciface_buddyTraceCycleEndPath(INT64_MAX, 1, INT64_MAX, 0, 0, 0);
  // Read through the hart after the timed interval, including dirty cache
  // lines.
  auto *stream = reinterpret_cast<volatile uint32_t *>(0x60000000UL);
  *stream = 0x42424f55;
  for (uintptr_t offset = 0; offset < sizeof(gvn_out); offset += 4) {
    uint32_t word = 0;
    const uintptr_t left = sizeof(gvn_out) - offset;
    __builtin_memcpy(&word, gvn_out + offset, left < 4 ? left : 4);
    *stream = word;
  }
  return 0;
}

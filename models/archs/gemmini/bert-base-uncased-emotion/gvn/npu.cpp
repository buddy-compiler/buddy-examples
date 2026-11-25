// A GVN slice of BERT (emotion) on Gemmini: one operation Gemmini takes, on its
// GPU operands (gvn_in, at the offsets of gvn_table). BEMU captures the output;
// the generated ELF checks that reference on RTL.
#include <stdint.h>

alignas(64) static const unsigned char expected[] = {
#embed "gvn_expect.bin"
};

extern "C" {
alignas(64) unsigned char gvn_in[] = {
#embed "gvn_in.bin"
};
alignas(64) unsigned char gvn_table[] = {
#embed "gvn_table.bin"
};
// Loader initialization avoids guest clearing or copying the check buffer.
alignas(64) __attribute__((
    section(".data"))) unsigned char gvn_out[sizeof(expected)] = {
#embed "gvn_expect.bin"
};
void gvn_slice();
void _mlir_ciface_buddyTraceCycleStartPath(int64_t, int64_t, int64_t, int64_t,
                                           int64_t, int64_t);
void _mlir_ciface_buddyTraceCycleEndPath(int64_t, int64_t, int64_t, int64_t,
                                         int64_t, int64_t);
}

// BEMU captures the reference; the FPGA runner checks the streamed output.
__attribute__((used, section(".bbsim.capture_output"),
               aligned(8))) static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check = {gvn_out, expected, sizeof(expected)};

int main() {
  _mlir_ciface_buddyTraceCycleStartPath(INT64_MAX, 1, INT64_MAX, 0, 0, 0);
  gvn_slice();
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

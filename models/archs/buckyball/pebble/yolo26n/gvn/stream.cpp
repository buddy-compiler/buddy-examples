#include <stdint.h>

extern "C" {
extern unsigned char gvn_out[];
alignas(64) extern const unsigned char gvn_expected[] = {
#embed "gvn_expect.bin"
};
}

__attribute__((used, section(".bbsim.capture_output"),
               aligned(8))) static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check = {gvn_out, gvn_expected, sizeof(gvn_expected)};

void read_output() {
  auto *stream = reinterpret_cast<volatile uint32_t *>(0x60000000UL);
  *stream = 0x42424f55;
  for (uintptr_t offset = 0; offset < sizeof(gvn_expected); offset += 4) {
    uint32_t word = 0;
    const uintptr_t left = sizeof(gvn_expected) - offset;
    __builtin_memcpy(&word, gvn_out + offset, left < 4 ? left : 4);
    *stream = word;
  }
}

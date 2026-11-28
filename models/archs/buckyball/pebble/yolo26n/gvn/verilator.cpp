#include <algorithm>
#include <bbhw/isa/isa.h>

extern "C" {
extern unsigned char gvn_out[];
alignas(64) extern const unsigned char gvn_expected[] = {
#embed "gvn_expect.bin"
};
// The driver fills this separate destination with the complement of the
// captured BEMU bytes before running the checked ELF on RTL.
alignas(64) unsigned char gvn_readback[sizeof(gvn_expected)] = {1};
}
static_assert(sizeof(gvn_readback) % 16 == 0);

__attribute__((used, section(".bbsim.check"), aligned(8))) static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check = {gvn_readback, gvn_expected, sizeof(gvn_expected)};

void read_output() {
  // Copy through the accelerator's coherent memory path after the timed ROI.
  static_assert(HART_ACCELERATOR_MEM_BANK_WIDTH == 128);
  constexpr uintptr_t chunk_bytes =
      std::min<uintptr_t>(4096, HART_ACCELERATOR_MEM_BANK_ENTRIES * 16);
  static_assert(chunk_bytes > 0);
  bb_mem_alloc(0, 1, 1);
  for (uintptr_t offset = 0; offset < sizeof(gvn_readback);
       offset += chunk_bytes) {
    uintptr_t bytes = sizeof(gvn_readback) - offset;
    if (bytes > chunk_bytes)
      bytes = chunk_bytes;
    bb_mvin(reinterpret_cast<uintptr_t>(gvn_out + offset), 0, bytes / 16, 1);
    bb_mvout(reinterpret_cast<uintptr_t>(gvn_readback + offset), 0, bytes / 16,
             1);
  }
  bb_mem_release(0);
  bb_barrier();
}

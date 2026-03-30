#include <stdint.h>

static inline uint64_t read_cycles() {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}

extern "C" double rtclock() {
  return static_cast<double>(read_cycles());
}

extern "C" double _mlir_ciface_rtclock() { return rtclock(); }

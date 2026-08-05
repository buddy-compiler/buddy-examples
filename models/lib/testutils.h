#ifndef TESTUTILS_H
#define TESTUTILS_H

#include <stdint.h>

#if defined(__riscv)
static inline uint64_t read_cycles() {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}
#else
#include <chrono>
static inline uint64_t read_cycles() {
  using clock = std::chrono::steady_clock;
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          clock::now().time_since_epoch())
          .count());
}
#endif

#endif // TESTUTILS_H

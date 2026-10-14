#ifndef TESTUTILS_H
#define TESTUTILS_H

#include <stdint.h>

static inline uint64_t read_counter() {
#if defined(__riscv)
  uint64_t value;
  asm volatile("rdcycle %0" : "=r"(value)::"memory");
  return value;
#elif defined(__x86_64__) && defined(__linux__)
  uint32_t low, high;
  asm volatile("lfence; rdtsc" : "=a"(low), "=d"(high)::"memory");
  return (uint64_t(high) << 32) | low;
#else
#error "Benchmark counter source is unsupported"
#endif
}

#endif // TESTUTILS_H

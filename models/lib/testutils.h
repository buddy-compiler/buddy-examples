#ifndef TESTUTILS_H
#define TESTUTILS_H

#include <stdint.h>
#if defined(__linux__)
#include <stdlib.h>
#include <time.h>
#endif

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

#if defined(__linux__)
// Wall-clock measurements retain time units; cycle counters are not seconds.
static inline uint64_t read_time_ns() {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value))
    abort();
  return (uint64_t)value.tv_sec * 1000000000ULL + (uint64_t)value.tv_nsec;
}
#endif

#endif // TESTUTILS_H

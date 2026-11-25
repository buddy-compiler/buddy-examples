#ifndef TESTUTILS_H
#define TESTUTILS_H

#include <stdint.h>
#include <stdio.h>
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

// A non-negative value with nine decimals; newlib-nano printf has no %f.
static inline void print_decimal(float value) {
  if (!(value < 4e9f)) {
    printf(value == value ? "inf" : "nan");
    return;
  }
  const uint64_t nano = (uint64_t)((double)value * 1e9 + 0.5);
  printf("%lu.%09lu", (unsigned long)(nano / 1000000000),
         (unsigned long)(nano % 1000000000));
}

// The final output of a model against the reference of its recipe. Prints
// "Output max diff: <max |actual - expected|> (<n> elements, relative
// <max diff / max |expected|>)" and returns nonzero when the relative
// difference is above `tolerance` or not a number.
static inline int check_output(const float *actual, const float *expected,
                               unsigned long count, float tolerance) {
  float largest = 0, scale = 0;
  for (unsigned long i = 0; i < count; ++i) {
    const float difference = actual[i] > expected[i] ? actual[i] - expected[i]
                                                     : expected[i] - actual[i];
    const float magnitude = expected[i] < 0 ? -expected[i] : expected[i];
    // A NaN stays the result once it appears.
    if (largest == largest && !(difference <= largest))
      largest = difference;
    if (magnitude > scale)
      scale = magnitude;
  }
  const float relative = largest / scale;
  printf("Output max diff: ");
  print_decimal(largest);
  printf(" (%lu elements, relative ", count);
  print_decimal(relative);
  printf(")\n");
  return !(relative <= tolerance);
}

#endif // TESTUTILS_H

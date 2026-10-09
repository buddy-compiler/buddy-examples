#pragma once
#include <stdint.h>
#include <stdlib.h>
#if defined(__linux__)
#include <sched.h>
#endif

static inline uint32_t runtime_cpu_id(void) {
#if defined(__linux__)
  int cpu = sched_getcpu();
  if (cpu < 0)
    abort();
  return (uint32_t)cpu;
#elif defined(__riscv)
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart > UINT32_MAX)
    abort();
  return (uint32_t)hart;
#else
#error "Runtime core location requires Linux CPU identity or RISC-V mhartid"
#endif
}

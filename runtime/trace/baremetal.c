#include "event.h"
#include "stream.h"
#include <stdint.h>

static void emit(uint64_t event, int64_t id, int64_t depth, int64_t p0,
                 int64_t p1, int64_t p2, int64_t p3) {
  uintptr_t hart;
  uint64_t cycle;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  __asm__ volatile("csrr %0, mcycle" : "=r"(cycle)::"memory");
  const struct trace_event record = {event, id,   hart,  stream_thread(),
                                     0,     hart, depth, {p0, p1, p2, p3},
                                     cycle};
  volatile uint64_t *port =
      (volatile uint64_t *)(DEVICE_TRACE_BASE + hart * DEVICE_TRACE_STRIDE);
  const unsigned long long *words = (const unsigned long long *)&record;
  for (unsigned i = 0; i < TRACE_WORDS; ++i) {
    port[i] = words[i];
    __asm__ volatile("fence iorw, iorw" ::: "memory");
  }
}

void _mlir_ciface_buddyTraceCycleStartPath(int64_t id, int64_t depth,
                                           int64_t p0, int64_t p1, int64_t p2,
                                           int64_t p3) {
  emit(TRACE_BEGIN, id, depth, p0, p1, p2, p3);
}

void _mlir_ciface_buddyTraceCycleEndPath(int64_t id, int64_t depth, int64_t p0,
                                         int64_t p1, int64_t p2, int64_t p3) {
  emit(TRACE_END, id, depth, p0, p1, p2, p3);
}

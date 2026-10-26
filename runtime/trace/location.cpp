#include "location.h"
#include "stream.h"
#include <cstdlib>
#if defined(__linux__)
#include <sched.h>
#endif

thread_location_t runtime_trace::location() {
#if defined(__linux__)
  int cpu = sched_getcpu();
  if (cpu < 0)
    abort();
  return {static_cast<uint32_t>(cpu), stream_thread()};
#else
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  return {static_cast<uint32_t>(hart), stream_thread()};
#endif
}

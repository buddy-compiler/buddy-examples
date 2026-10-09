#include "runtime.h"
#include <cstdlib>
#include <sched.h>

static thread_local int controller = -1;
void native_trace_bind() {
  if (controller >= 0) {
    cpu_set_t affinity;
    if (sched_getaffinity(0, sizeof(affinity), &affinity) ||
        CPU_COUNT(&affinity) != 1 || !CPU_ISSET(controller, &affinity))
      std::abort();
    return;
  }
  int cpu = sched_getcpu();
  if (cpu < 0)
    std::abort();
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  CPU_SET(cpu, &affinity);
  if (sched_setaffinity(0, sizeof(affinity), &affinity))
    std::abort();
  controller = cpu;
}
extern "C" core_location_t core_location() {
  int cpu = sched_getcpu();
  if (cpu < 0)
    std::abort();
  return {static_cast<uint32_t>(cpu), 0};
}

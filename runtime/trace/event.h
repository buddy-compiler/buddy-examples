#pragma once

enum { TRACE_WORDS = 12, TRACE_BEGIN = 1, TRACE_END = 2 };

// Twelve consecutive 64-bit MMIO writes; the cycle word commits the event.
struct trace_event {
  unsigned long long event, id, controller, thread, pid, tid, depth;
  unsigned long long path[4];
  unsigned long long cycle;
};

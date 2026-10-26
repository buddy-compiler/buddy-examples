#pragma once
#include <cstddef>
#include <cstdint>

/* Where a trace event happened: the core and the hardware thread executing its
 * current instruction stream (0 on the core itself). */
typedef struct {
  uint32_t controller;
  size_t thread;
} thread_location_t;

namespace runtime_trace {
thread_location_t location();
}

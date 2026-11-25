// Cycle traces of a program on the device under Linux: each begin and end
// goes to /dev/trace, which records them with the device's other events.
#include "event.h"
#include "trace.h"
#include <fcntl.h>
#include <unistd.h>

static void emitCycle(uint64_t event, int64_t id, int64_t depth, int64_t p0,
                      int64_t p1, int64_t p2, int64_t p3) {
  auto location = runtime_trace::location();
  auto identity = processThread();
  trace_event record{event,
                     static_cast<uint64_t>(id),
                     location.controller,
                     location.thread,
                     identity.first,
                     identity.second,
                     static_cast<uint64_t>(depth),
                     {static_cast<uint64_t>(p0), static_cast<uint64_t>(p1),
                      static_cast<uint64_t>(p2), static_cast<uint64_t>(p3)},
                     readCycle()};
  static const int device = open("/dev/trace", O_WRONLY | O_CLOEXEC);
  if (device < 0 || write(device, &record, sizeof(record)) != sizeof(record))
    traceError("cannot submit cycle event to /dev/trace");
}

void beginCycle(int64_t id, int64_t depth, int64_t p0, int64_t p1, int64_t p2,
                int64_t p3) {
  emitCycle(TRACE_BEGIN, id, depth, p0, p1, p2, p3);
}
void endCycle(int64_t id, int64_t depth, int64_t p0, int64_t p1, int64_t p2,
              int64_t p3) {
  emitCycle(TRACE_END, id, depth, p0, p1, p2, p3);
}

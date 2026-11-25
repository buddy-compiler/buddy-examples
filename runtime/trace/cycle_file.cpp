// Cycle traces of a native program: each matched begin and end is written
// next to the program, with a summary per controller and thread.
#include "trace.h"
#include <array>
#include <map>
#include <mutex>
#include <set>
#include <string>

struct TraceStart {
  bool active = false;
  uint64_t cycle = 0, pid = 0, tid = 0;
  std::string path;
};
struct TraceState {
  std::array<TraceStart, kTraceMaxId> starts;
  std::set<std::string> written;
  uint64_t first = 0, last = 0, sum = 0, count = 0;
  bool began = false;
  ~TraceState() {
    for (const auto &start : starts)
      if (start.active)
        traceError("process exited with an unmatched start");
  }
};
static std::map<std::pair<uint32_t, size_t>, TraceState> traceStates;

void beginCycle(int64_t id, int64_t depth, int64_t p0, int64_t p1, int64_t p2,
                int64_t p3) {
  if (id < 0 || id >= kTraceMaxId)
    traceError("ID out of range");
  auto location = runtime_trace::location();
  auto identity = processThread();
  auto cycle = readCycle();
  char key[64];
  writeTracePath(key, sizeof(key), depth, p0, p1, p2, p3);
  std::lock_guard<std::mutex> guard(traceMutex);
  auto &state = traceStates[{location.controller, location.thread}];
  auto &start = state.starts[id];
  if (start.active)
    traceError("start already active in this controller/thread");
  start = {true, cycle, identity.first, identity.second, key};
  if (!state.began) {
    state.first = cycle;
    state.began = true;
  }
}
void endCycle(int64_t id, int64_t depth, int64_t p0, int64_t p1, int64_t p2,
              int64_t p3) {
  if (id < 0 || id >= kTraceMaxId)
    traceError("ID out of range");
  auto location = runtime_trace::location();
  auto identity = processThread();
  auto end = readCycle();
  char key[64];
  writeTracePath(key, sizeof(key), depth, p0, p1, p2, p3);
  std::lock_guard<std::mutex> guard(traceMutex);
  auto &state = traceStates[{location.controller, location.thread}];
  auto &start = state.starts[id];
  if (!start.active || start.pid != identity.first ||
      start.tid != identity.second || start.path != key)
    traceError(
        "end has no matching start in this thread and controller/thread");
  if (end < start.cycle)
    traceError("counter moved backwards");
  auto cycle = end - start.cycle;
  FILE *file = openTraceFilePath("cycle", depth, p0, p1, p2, p3,
                                 state.written.count(key) ? "a" : "w");
  std::fprintf(file,
               "platform %s\ncounter %s\npid %llu\ntid %llu\nstart %llu\nend "
               "%llu\nelapsed %llu\n",
               platformName(), counterName(), (unsigned long long)start.pid,
               (unsigned long long)start.tid, (unsigned long long)start.cycle,
               (unsigned long long)end, (unsigned long long)cycle);
  if (std::fclose(file))
    traceError("cannot close cycle record");
  state.written.insert(key);
  start.active = false;
  state.last = end;
  if (depth == 1)
    state.sum += cycle;
  ++state.count;
  auto path = traceRoot(location) / "cycle" / "summary.txt";
  file = std::fopen(path.c_str(), "w");
  if (!file)
    traceError("cannot open cycle summary");
  std::fprintf(
      file,
      "platform %s\ncounter %s\nfirst_start %llu\nlast_end %llu\ntrace_span "
      "%llu\ntraced_cycle_sum %llu\ntrace_count %llu\n",
      platformName(), counterName(), (unsigned long long)state.first,
      (unsigned long long)state.last,
      (unsigned long long)(state.last - state.first),
      (unsigned long long)state.sum, (unsigned long long)state.count);
  if (std::fclose(file))
    traceError("cannot close cycle summary");
}

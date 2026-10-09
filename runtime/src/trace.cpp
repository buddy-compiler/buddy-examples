#include "trace_internal.h"
#include <CRunnerUtils.h>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <string>
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

static void traceError(const char *message);
static constexpr int64_t kTraceMaxId = 4096;
static constexpr int64_t kTraceMaxPathDepth = 4;
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
static std::mutex traceMutex;
static std::map<std::pair<uint32_t, size_t>, TraceState> traceStates;

static void traceError(const char *message) {
  std::fprintf(stderr, "trace error: %s\n", message);
  std::abort();
}
static uint64_t readCycle() {
#if defined(__riscv)
  uint64_t cycle;
  asm volatile("rdcycle %0" : "=r"(cycle)::"memory");
  return cycle;
#elif defined(__x86_64__) && defined(__linux__)
  uint32_t lo, hi;
  asm volatile("lfence; rdtsc" : "=a"(lo), "=d"(hi)::"memory");
  return (uint64_t(hi) << 32) | lo;
#else
#error "Trace requires RISC-V cycle or Linux x86 TSC"
#endif
}
static const char *counterName() {
#if defined(__riscv)
  return "riscv-cycle";
#elif defined(__x86_64__) && defined(__linux__)
  return "x86-tsc";
#else
#error "Trace counter source is unsupported"
#endif
}
static const char *platformName() {
#if defined(__linux__)
  return "linux";
#elif defined(__riscv)
  return "baremetal";
#else
#error "Trace platform is unsupported"
#endif
}
static std::pair<uint64_t, uint64_t> processThread() {
#if defined(__linux__)
  auto pid = getpid();
  auto tid = syscall(SYS_gettid);
  if (pid <= 0 || tid <= 0)
    traceError("cannot query Linux PID/TID");
  return {uint64_t(pid), uint64_t(tid)};
#elif defined(__riscv)
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  return {0, uint64_t(hart)};
#else
#error "Trace process/thread identity is unsupported"
#endif
}
static std::filesystem::path traceRoot(core_location_t location) {
  return std::filesystem::path("trace") /
         ("controller-" + std::to_string(location.controller)) /
         ("core-" + std::to_string(location.core));
}
static uint64_t fnv1a64(const int8_t *data, size_t size, uint64_t hash) {
  for (size_t i = 0; i < size; ++i) {
    hash ^= static_cast<uint8_t>(data[i]);
    hash *= 1099511628211ULL;
  }
  return hash;
}
static void writeTracePath(char *buffer, size_t size, int64_t depth,
                           int64_t path0, int64_t path1, int64_t path2,
                           int64_t path3) {
  int64_t path[kTraceMaxPathDepth] = {path0, path1, path2, path3};
  if (depth <= 0 || depth > kTraceMaxPathDepth)
    traceError("path depth out of range");
  size_t offset = 0;
  for (int64_t i = 0; i < depth; ++i) {
    if (path[i] < 0)
      traceError("negative path component");
    int written = std::snprintf(buffer + offset, size - offset,
                                i ? "-%lld" : "%lld", (long long)path[i]);
    if (written < 0 || size_t(written) >= size - offset)
      traceError("path too long");
    offset += size_t(written);
  }
}
static FILE *openTraceFilePath(const char *kind, int64_t depth, int64_t path0,
                               int64_t path1, int64_t path2, int64_t path3,
                               const char *mode = "w") {
  char key[64];
  writeTracePath(key, sizeof(key), depth, path0, path1, path2, path3);
  auto directory = traceRoot(runtime_trace::location()) / kind;
  std::filesystem::create_directories(directory);
  auto path = directory / ("trace-" + std::string(key) + ".txt");
  FILE *file = std::fopen(path.c_str(), mode);
  if (!file || setvbuf(file, nullptr, _IONBF, 0))
    traceError("cannot open trace file");
  return file;
}
static FILE *openTraceFile(const char *kind, int64_t id) {
  return openTraceFilePath(kind, 1, id, -1, -1, -1);
}
static float bf16ToF32(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}
static void checkTraceTensor(void *tensor) {
  if (!tensor)
    traceError("null tensor");
}

extern "C" void
_mlir_ciface_buddyTraceTensorF32(int64_t id,
                                 StridedMemRefType<float, 1> *tensor) {
  std::lock_guard<std::mutex> guard(traceMutex);
  checkTraceTensor(tensor);

  DynamicMemRefType<float> ref(*tensor);

  FILE *file = openTraceFile("tensor", id);
  for (int64_t i = 0; i < ref.sizes[0]; ++i)
    fprintf(file, "%.9g\n", ref.data[ref.offset + i * ref.strides[0]]);
  fclose(file);
}

extern "C" void _mlir_ciface_buddyTraceTensorF32Path(
    int64_t id, int64_t depth, int64_t path0, int64_t path1, int64_t path2,
    int64_t path3, StridedMemRefType<float, 1> *tensor) {
  std::lock_guard<std::mutex> guard(traceMutex);
  (void)id;
  checkTraceTensor(tensor);

  DynamicMemRefType<float> ref(*tensor);

  FILE *file = openTraceFilePath("tensor", depth, path0, path1, path2, path3);
  for (int64_t i = 0; i < ref.sizes[0]; ++i)
    fprintf(file, "%.9g\n", ref.data[ref.offset + i * ref.strides[0]]);
  fclose(file);
}

extern "C" void
_mlir_ciface_buddyTraceTensorBF16(int64_t id,
                                  StridedMemRefType<uint16_t, 1> *tensor) {
  std::lock_guard<std::mutex> guard(traceMutex);
  checkTraceTensor(tensor);

  DynamicMemRefType<uint16_t> ref(*tensor);

  FILE *file = openTraceFile("tensor", id);
  for (int64_t i = 0; i < ref.sizes[0]; ++i) {
    uint16_t value = ref.data[ref.offset + i * ref.strides[0]];
    fprintf(file, "%.9g\n", bf16ToF32(value));
  }
  fclose(file);
}

extern "C" void _mlir_ciface_buddyTraceTensorBF16Path(
    int64_t id, int64_t depth, int64_t path0, int64_t path1, int64_t path2,
    int64_t path3, StridedMemRefType<uint16_t, 1> *tensor) {
  std::lock_guard<std::mutex> guard(traceMutex);
  (void)id;
  checkTraceTensor(tensor);

  DynamicMemRefType<uint16_t> ref(*tensor);

  FILE *file = openTraceFilePath("tensor", depth, path0, path1, path2, path3);
  for (int64_t i = 0; i < ref.sizes[0]; ++i) {
    uint16_t value = ref.data[ref.offset + i * ref.strides[0]];
    fprintf(file, "%.9g\n", bf16ToF32(value));
  }
  fclose(file);
}

extern "C" void
_mlir_ciface_buddyTraceTensorI8Path(int64_t id, int64_t depth, int64_t path0,
                                    int64_t path1, int64_t path2, int64_t path3,
                                    StridedMemRefType<int8_t, 1> *tensor) {
  std::lock_guard<std::mutex> guard(traceMutex);
  (void)id;
  checkTraceTensor(tensor);

  DynamicMemRefType<int8_t> ref(*tensor);
  FILE *file = openTraceFilePath("tensor", depth, path0, path1, path2, path3);
  for (int64_t i = 0; i < ref.sizes[0]; ++i)
    fprintf(file, "%d\n",
            static_cast<int>(ref.data[ref.offset + i * ref.strides[0]]));
  fclose(file);
}

extern "C" void _mlir_ciface_buckyballTraceStageI8Path(
    int64_t id, int64_t depth, int64_t path0, int64_t path1, int64_t path2,
    int64_t path3, StridedMemRefType<int8_t, 1> *tensor) {
  std::lock_guard<std::mutex> guard(traceMutex);
  (void)id;
  checkTraceTensor(tensor);

  DynamicMemRefType<int8_t> ref(*tensor);
  if (ref.strides[0] != 1) {
    fprintf(stderr, "Buckyball stage trace requires a contiguous tensor\n");
    abort();
  }

  char key[64];
  writeTracePath(key, sizeof(key), depth, path0, path1, path2, path3);
  size_t size = static_cast<size_t>(ref.sizes[0]);
  constexpr size_t chunkSize = 8192;
  int8_t buffer[chunkSize];
  uint64_t hash = 1469598103934665603ULL;
  for (size_t offset = 0, part = 0; offset < size; ++part) {
    size_t chunk = std::min(chunkSize, size - offset);
    memcpy(buffer, ref.data + ref.offset + offset, chunk);
    hash = fnv1a64(buffer, chunk, hash);
    auto directory = traceRoot(runtime_trace::location()) / "tensor";
    std::filesystem::create_directories(directory);
    auto path = directory / ("trace-" + std::string(key) + "-part-" +
                             std::to_string(part) + ".i8");
    FILE *file = fopen(path.c_str(), "wb");
    if (!file) {
      fprintf(stderr, "failed to open stage trace file: %s: %s\n", path.c_str(),
              strerror(errno));
      abort();
    }
    if (fwrite(buffer, 1, chunk, file) != chunk) {
      fprintf(stderr, "failed to write complete stage trace: %s: %s\n",
              path.c_str(), strerror(errno));
      abort();
    }
    if (fclose(file) != 0) {
      fprintf(stderr, "failed to close stage trace file: %s: %s\n",
              path.c_str(), strerror(errno));
      abort();
    }
    offset += chunk;
  }
  auto location = runtime_trace::location();
  auto identity = processThread();
  fprintf(stdout,
          "STAGE_TRACE controller=%u core=%zu platform=%s pid=%llu tid=%llu "
          "path=%s size=%zu hash=%016llx\n",
          location.controller, location.core, platformName(),
          (unsigned long long)identity.first,
          (unsigned long long)identity.second, key, size,
          (unsigned long long)hash);
  fflush(stdout);
}

extern "C" void _mlir_ciface_buddyTraceTensorI32Path(
    int64_t id, int64_t depth, int64_t path0, int64_t path1, int64_t path2,
    int64_t path3, StridedMemRefType<int32_t, 1> *tensor) {
  std::lock_guard<std::mutex> guard(traceMutex);
  (void)id;
  checkTraceTensor(tensor);

  DynamicMemRefType<int32_t> ref(*tensor);
  FILE *file = openTraceFilePath("tensor", depth, path0, path1, path2, path3);
  for (int64_t i = 0; i < ref.sizes[0]; ++i)
    fprintf(file, "%d\n",
            static_cast<int>(ref.data[ref.offset + i * ref.strides[0]]));
  fclose(file);
}

static void beginCycle(int64_t id, int64_t depth, int64_t p0, int64_t p1,
                       int64_t p2, int64_t p3) {
  if (id < 0 || id >= kTraceMaxId)
    traceError("ID out of range");
  auto location = runtime_trace::location();
  auto identity = processThread();
  auto cycle = readCycle();
  char key[64];
  writeTracePath(key, sizeof(key), depth, p0, p1, p2, p3);
  std::lock_guard<std::mutex> guard(traceMutex);
  auto &state = traceStates[{location.controller, location.core}];
  auto &start = state.starts[id];
  if (start.active)
    traceError("start already active in this controller/core");
  start = {true, cycle, identity.first, identity.second, key};
  if (!state.began) {
    state.first = cycle;
    state.began = true;
  }
}
static void endCycle(int64_t id, int64_t depth, int64_t p0, int64_t p1,
                     int64_t p2, int64_t p3) {
  if (id < 0 || id >= kTraceMaxId)
    traceError("ID out of range");
  auto location = runtime_trace::location();
  auto identity = processThread();
  auto end = readCycle();
  char key[64];
  writeTracePath(key, sizeof(key), depth, p0, p1, p2, p3);
  std::lock_guard<std::mutex> guard(traceMutex);
  auto &state = traceStates[{location.controller, location.core}];
  auto &start = state.starts[id];
  if (!start.active || start.pid != identity.first ||
      start.tid != identity.second || start.path != key)
    traceError("end has no matching start in this thread and controller/core");
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
extern "C" void _mlir_ciface_buddyTraceCycleStart(int64_t id) {
  beginCycle(id, 1, id, -1, -1, -1);
}
extern "C" void _mlir_ciface_buddyTraceCycleStartPath(int64_t id, int64_t depth,
                                                      int64_t p0, int64_t p1,
                                                      int64_t p2, int64_t p3) {
  beginCycle(id, depth, p0, p1, p2, p3);
}
extern "C" void _mlir_ciface_buddyTraceCycleEnd(int64_t id) {
  endCycle(id, 1, id, -1, -1, -1);
}
extern "C" void _mlir_ciface_buddyTraceCycleEndPath(int64_t id, int64_t depth,
                                                    int64_t p0, int64_t p1,
                                                    int64_t p2, int64_t p3) {
  endCycle(id, depth, p0, p1, p2, p3);
}

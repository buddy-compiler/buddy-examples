#include "trace.h"
#include "location.h"
#include <CRunnerUtils.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

std::mutex traceMutex;

void traceError(const char *message) {
  std::fprintf(stderr, "trace error: %s\n", message);
  std::abort();
}
uint64_t readCycle() {
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
const char *counterName() {
#if defined(__riscv)
  return "riscv-cycle";
#elif defined(__x86_64__) && defined(__linux__)
  return "x86-tsc";
#else
#error "Trace counter source is unsupported"
#endif
}
const char *platformName() {
#if defined(__linux__)
  return "linux";
#elif defined(__riscv)
  return "baremetal";
#else
#error "Trace platform is unsupported"
#endif
}
std::pair<uint64_t, uint64_t> processThread() {
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
// Traces go next to the program that writes them, wherever it is started
// from.
std::filesystem::path traceRoot(thread_location_t location) {
#if defined(__linux__)
  auto base = std::filesystem::read_symlink("/proc/self/exe").parent_path();
#else
  std::filesystem::path base;
#endif
  return base / "trace" /
         ("controller-" + std::to_string(location.controller)) /
         ("thread-" + std::to_string(location.thread));
}
void writeTracePath(char *buffer, size_t size, int64_t depth, int64_t path0,
                    int64_t path1, int64_t path2, int64_t path3) {
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
FILE *openTraceFilePath(const char *kind, int64_t depth, int64_t path0,
                        int64_t path1, int64_t path2, int64_t path3,
                        const char *mode) {
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
  return openTraceFilePath(kind, 1, id, -1, -1, -1, "w");
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

  FILE *file =
      openTraceFilePath("tensor", depth, path0, path1, path2, path3, "w");
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

  FILE *file =
      openTraceFilePath("tensor", depth, path0, path1, path2, path3, "w");
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
  FILE *file =
      openTraceFilePath("tensor", depth, path0, path1, path2, path3, "w");
  for (int64_t i = 0; i < ref.sizes[0]; ++i)
    fprintf(file, "%d\n",
            static_cast<int>(ref.data[ref.offset + i * ref.strides[0]]));
  fclose(file);
}

extern "C" void _mlir_ciface_buddyTraceTensorI32Path(
    int64_t id, int64_t depth, int64_t path0, int64_t path1, int64_t path2,
    int64_t path3, StridedMemRefType<int32_t, 1> *tensor) {
  std::lock_guard<std::mutex> guard(traceMutex);
  (void)id;
  checkTraceTensor(tensor);

  DynamicMemRefType<int32_t> ref(*tensor);
  FILE *file =
      openTraceFilePath("tensor", depth, path0, path1, path2, path3, "w");
  for (int64_t i = 0; i < ref.sizes[0]; ++i)
    fprintf(file, "%d\n",
            static_cast<int>(ref.data[ref.offset + i * ref.strides[0]]));
  fclose(file);
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

// The code here is from the upstream llvm, you can check
// llvm/mlir/include/mlir/ExecutionEngine/CRunnerUtils.cpp

#include <CRunnerUtils.h>
#include <Msan.h>
#include <algorithm>
#include <alloca.h>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <malloc.h>
#include <string.h>

static constexpr int64_t kTraceMaxId = 4096;
static uint64_t traceCycleStart[kTraceMaxId];
static uint64_t traceFirstCycle = 0;
static uint64_t traceLastCycle = 0;
static uint64_t traceCycleSum = 0;
static uint64_t traceCycleCount = 0;

static uint64_t readCycle() {
#if defined(__riscv)
  uint64_t cycle = 0;
  asm volatile("rdcycle %0" : "=r"(cycle));
  return cycle;
#else
  return 0;
#endif
}

static FILE *openTraceFile(const char *kind, int64_t id) {
  char path[128];
  snprintf(path, sizeof(path), "trace/%s/trace-%lld.txt", kind,
           static_cast<long long>(id));
  FILE *file = fopen(path, "w");
  if (!file) {
    fprintf(stderr, "failed to open trace file: %s\n", path);
    abort();
  }
  return file;
}

static float bf16ToF32(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

static void writeCycleSummary() {
  FILE *file = fopen("trace/cycle/summary.txt", "w");
  if (!file) {
    fprintf(stderr, "failed to open trace file: trace/cycle/summary.txt\n");
    abort();
  }
  fprintf(file, "first_start %llu\n",
          static_cast<unsigned long long>(traceFirstCycle));
  fprintf(file, "last_end %llu\n",
          static_cast<unsigned long long>(traceLastCycle));
  fprintf(file, "trace_span %llu\n",
          static_cast<unsigned long long>(traceLastCycle - traceFirstCycle));
  fprintf(file, "traced_cycle_sum %llu\n",
          static_cast<unsigned long long>(traceCycleSum));
  fprintf(file, "trace_count %llu\n",
          static_cast<unsigned long long>(traceCycleCount));
  fclose(file);
}

static void checkTraceTensor(void *tensor) {
  if (!tensor) {
    fprintf(stderr, "trace tensor is null\n");
    abort();
  }
}

extern "C" void memrefCopy(int64_t elemSize, UnrankedMemRefType<char> *srcArg,
                           UnrankedMemRefType<char> *dstArg) {
  DynamicMemRefType<char> src(*srcArg);
  DynamicMemRefType<char> dst(*dstArg);

  int64_t rank = src.rank;
  MLIR_MSAN_MEMORY_IS_INITIALIZED(src.sizes, rank * sizeof(int64_t));

  // Handle empty shapes -> nothing to copy.
  for (int rankp = 0; rankp < rank; ++rankp)
    if (src.sizes[rankp] == 0)
      return;

  char *srcPtr = src.data + src.offset * elemSize;
  char *dstPtr = dst.data + dst.offset * elemSize;

  if (rank == 0) {
    memcpy(dstPtr, srcPtr, elemSize);
    return;
  }

  int64_t *indices = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));
  int64_t *srcStrides = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));
  int64_t *dstStrides = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));

  // Initialize index and scale strides.
  for (int rankp = 0; rankp < rank; ++rankp) {
    indices[rankp] = 0;
    srcStrides[rankp] = src.strides[rankp] * elemSize;
    dstStrides[rankp] = dst.strides[rankp] * elemSize;
  }

  int64_t readIndex = 0, writeIndex = 0;
  for (;;) {
    // Copy over the element, byte by byte.
    memcpy(dstPtr + writeIndex, srcPtr + readIndex, elemSize);
    // Advance index and read position.
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      // Advance at current axis.
      auto newIndex = ++indices[axis];
      readIndex += srcStrides[axis];
      writeIndex += dstStrides[axis];
      // If this is a valid index, we have our next index, so continue copying.
      if (src.sizes[axis] != newIndex)
        break;
      // We reached the end of this axis. If this is axis 0, we are done.
      if (axis == 0)
        return;
      // Else, reset to 0 and undo the advancement of the linear index that
      // this axis had. Then continue with the axis one outer.
      indices[axis] = 0;
      readIndex -= src.sizes[axis] * srcStrides[axis];
      writeIndex -= dst.sizes[axis] * dstStrides[axis];
    }
  }
}

extern "C" void printF64(double d) { fprintf(stdout, "%lg", d); }

extern "C" void printNewline() { fputc('\n', stdout); }

extern "C" void
_mlir_ciface_buddyTraceTensorF32(int64_t id,
                                 StridedMemRefType<float, 1> *tensor) {
  checkTraceTensor(tensor);

  DynamicMemRefType<float> ref(*tensor);

  FILE *file = openTraceFile("tensor", id);
  for (int64_t i = 0; i < ref.sizes[0]; ++i)
    fprintf(file, "%.9g\n", ref.data[ref.offset + i * ref.strides[0]]);
  fclose(file);
}

extern "C" void
_mlir_ciface_buddyTraceTensorBF16(int64_t id,
                                  StridedMemRefType<uint16_t, 1> *tensor) {
  checkTraceTensor(tensor);

  DynamicMemRefType<uint16_t> ref(*tensor);

  FILE *file = openTraceFile("tensor", id);
  for (int64_t i = 0; i < ref.sizes[0]; ++i) {
    uint16_t value = ref.data[ref.offset + i * ref.strides[0]];
    fprintf(file, "%.9g\n", bf16ToF32(value));
  }
  fclose(file);
}

extern "C" void _mlir_ciface_buddyTraceCycleStart(int64_t id) {
  if (id < 0 || id >= kTraceMaxId) {
    fprintf(stderr, "trace id out of range: %lld\n",
            static_cast<long long>(id));
    abort();
  }
  uint64_t start = readCycle();
  if (traceCycleCount == 0)
    traceFirstCycle = start;
  traceCycleStart[id] = start;
}

extern "C" void _mlir_ciface_buddyTraceCycleEnd(int64_t id) {
  if (id < 0 || id >= kTraceMaxId) {
    fprintf(stderr, "trace id out of range: %lld\n",
            static_cast<long long>(id));
    abort();
  }
  uint64_t end = readCycle();
  uint64_t cycle = end - traceCycleStart[id];
  traceLastCycle = end;
  traceCycleSum += cycle;
  traceCycleCount += 1;

  FILE *file = openTraceFile("cycle", id);
  fprintf(file, "start %llu\n",
          static_cast<unsigned long long>(traceCycleStart[id]));
  fprintf(file, "end %llu\n", static_cast<unsigned long long>(end));
  fprintf(file, "elapsed %llu\n", static_cast<unsigned long long>(cycle));
  fclose(file);
  writeCycleSummary();
}

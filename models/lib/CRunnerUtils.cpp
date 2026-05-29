// The code here is from the upstream llvm, you can check
// llvm/mlir/include/mlir/ExecutionEngine/CRunnerUtils.cpp

#include <CRunnerUtils.h>
#include <Msan.h>
#include <algorithm>
#include <alloca.h>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <malloc.h>
#include <random>
#include <sstream>
#include <string>
#include <string.h>
#include <sys/time.h>
#include <vector>

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

namespace {
struct TraceMeta {
  const char *tag;
  const char *layout;
  std::vector<int64_t> shape;
};

const TraceMeta kTraceMeta[] = {
    {"input_nchw", "nchw", {1, 1, 28, 28}},
    {"conv1_out_nchw", "nchw", {1, 6, 24, 24}},
    {"relu1_out_nchw", "nchw", {1, 6, 24, 24}},
    {"pool1_out_nchw", "nchw", {1, 6, 12, 12}},
    {"conv2_out_nchw", "nchw", {1, 16, 8, 8}},
    {"relu2_out_nchw", "nchw", {1, 16, 8, 8}},
    {"pool2_out_nchw", "nchw", {1, 16, 4, 4}},
    {"flatten_out", "nc", {1, 256}},
    {"fc1_out", "nc", {1, 120}},
    {"relu3_out", "nc", {1, 120}},
    {"fc2_out", "nc", {1, 84}},
    {"relu4_out", "nc", {1, 84}},
    {"fc3_out", "nc", {1, 10}},
};

std::string escapeJson(const char *text) {
  std::ostringstream os;
  for (const char *p = text; p && *p; ++p) {
    switch (*p) {
    case '\\':
      os << "\\\\";
      break;
    case '"':
      os << "\\\"";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      os << *p;
      break;
    }
  }
  return os.str();
}

FILE *traceStream() {
  static FILE *stream = nullptr;
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    if (const char *path = std::getenv("BB_LAYER_TRACE_PATH"); path && *path) {
      stream = fopen(path, "w");
    } else {
      stream = fopen("layer-trace.ndjson", "w");
    }
  }
  return stream;
}

bool traceEnabled() { return traceStream() != nullptr; }
} // namespace

extern "C" void _mlir_ciface_buddyTraceTensorF32(
    int64_t tagId, StridedMemRefType<float, 1> *arg) {
  if (!traceEnabled())
    return;
  if (tagId < 0 || static_cast<size_t>(tagId) >= std::size(kTraceMeta))
    return;

  const TraceMeta &meta = kTraceMeta[tagId];

  FILE *stream = traceStream();
  fprintf(stream, "{\"tag\":\"%s\",\"layout\":\"%s\",\"shape\":[",
          escapeJson(meta.tag).c_str(), escapeJson(meta.layout).c_str());
  for (size_t i = 0; i < meta.shape.size(); ++i) {
    if (i != 0)
      fputc(',', stream);
    fprintf(stream, "%" PRId64, meta.shape[i]);
  }
  fputs("],\"values\":[", stream);
  for (int64_t i = 0; i < arg->sizes[0]; ++i) {
    if (i != 0)
      fputc(',', stream);
    fprintf(stream, "%.9g", arg->data[arg->offset + i * arg->strides[0]]);
  }
  fputs("]}\n", stream);
  fflush(stream);
}

extern "C" void buddyTraceTensorF32(const char *tag, const char *layout,
                                    int64_t rank, const int64_t *shape,
                                    const float *data, int64_t elemCount) {
  if (!traceEnabled())
    return;

  FILE *stream = traceStream();
  fprintf(stream, "{\"tag\":\"%s\",\"layout\":\"%s\",\"shape\":[",
          escapeJson(tag).c_str(), escapeJson(layout).c_str());
  for (int64_t i = 0; i < rank; ++i) {
    if (i != 0)
      fputc(',', stream);
    fprintf(stream, "%" PRId64, shape[i]);
  }
  fputs("],\"values\":[", stream);
  for (int64_t i = 0; i < elemCount; ++i) {
    if (i != 0)
      fputc(',', stream);
    fprintf(stream, "%.9g", data[i]);
  }
  fputs("]}\n", stream);
  fflush(stream);
}

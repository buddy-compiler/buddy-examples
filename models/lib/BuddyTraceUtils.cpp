#include "BuddyTraceUtils.h"
#include <CRunnerUtils.h>
#include <Msan.h>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

namespace {
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

[[noreturn]] void failTrace(const char *msg) {
  fprintf(stderr, "BuddyTrace error: %s\n", msg);
  std::abort();
}

FILE *traceStream() {
  static FILE *stream = nullptr;
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    const char *path = std::getenv("BB_LAYER_TRACE_PATH");
    if (!path || !*path)
      failTrace("BB_LAYER_TRACE_PATH is required");
    stream = fopen(path, "w");
    if (!stream)
      failTrace("failed to open BB_LAYER_TRACE_PATH");
  }
  return stream;
}

const BuddyTraceMeta &getMetaById(int64_t tagId) {
  size_t count = 0;
  const BuddyTraceMeta *meta = buddyGetTraceMeta(&count);
  if (!meta)
    failTrace("buddyGetTraceMeta returned null");
  if (tagId < 0 || static_cast<size_t>(tagId) >= count)
    failTrace("trace tag id out of range");
  return meta[tagId];
}

void writeTrace(const char *tag, const char *layout, int64_t rank,
                const int64_t *shape, const float *data, int64_t elemCount,
                int64_t stride) {
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
    fprintf(stream, "%.9g", data[i * stride]);
  }
  fputs("]}\n", stream);
  fflush(stream);
}
} // namespace

extern "C" void _mlir_ciface_buddyTraceTensorF32(
    int64_t tagId, StridedMemRefType<float, 1> *arg) {
  const BuddyTraceMeta &meta = getMetaById(tagId);
  MLIR_MSAN_MEMORY_IS_INITIALIZED(arg->sizes, sizeof(int64_t));
  writeTrace(meta.tag, meta.layout, meta.rank, meta.shape,
             arg->data + arg->offset, arg->sizes[0], arg->strides[0]);
}

extern "C" void buddyTraceTensorF32(const char *tag, const char *layout,
                                    int64_t rank, const int64_t *shape,
                                    const float *data, int64_t elemCount) {
  writeTrace(tag, layout, rank, shape, data, elemCount, 1);
}

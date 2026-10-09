#pragma once
#include "weights.h"
#include <atomic>
#include <buddy/Core/Container.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <sys/types.h>
#include <vector>

using Hidden = MemRef<float, 3>;
using Cache = MemRef<int8_t, 4>;
using Floats = MemRef<float, 1>;
using Bytes = MemRef<int8_t, 1>;
using Tokens = MemRef<int64_t, 2>;
using Positions = MemRef<int64_t, 1>;

struct AttentionResult {
  Hidden hidden;
  Cache keyCodes, keyScales, valueCodes, valueScales;
};
struct Parameters {
  const char *directory;
  size_t floats, bytes;
};
struct EmbeddingEntry {
  Parameters parameters;
  void (*run)(Hidden *, Bytes *, Tokens *);
};
struct AttentionEntry {
  Parameters parameters;
  void (*run)(AttentionResult *, Floats *, Bytes *, Hidden *, Cache *, Cache *,
              Cache *, Cache *, Positions *);
};
struct FfnEntry {
  Parameters parameters;
  void (*run)(Hidden *, Floats *, Bytes *, Hidden *);
};
using OutputEntry = FfnEntry;

struct ModelShape {
  size_t hidden, layers, kvHeads, context, projection, parts, tiles,
      intermediate;
  size_t head, vocabulary, prefill, cache;
};
const ModelShape &modelShape();
int controlCpu(size_t tile);

struct Command {
  std::atomic<int> state;
  uint64_t operation, count, start, layer;
  char error[256];
};
struct Worker {
  Command *command;
  float *input, *output;
  int8_t *keyCodes, *keyScales, *valueCodes, *valueScales;
  pid_t pid;
  int notify[2], completion[2];
};
class Execution {
  void *storage;
  size_t storageBytes;
  bool closed = false;

public:
  int64_t *tokens;
  float *hidden;
  std::vector<Worker> workers;
  Execution(const std::filesystem::path &directory,
            const std::vector<uint64_t> &tiles);
  ~Execution() noexcept(false);
  void close();
  void submit(size_t rank, uint64_t operation, size_t count, size_t start,
              size_t layer);
  void wait(size_t rank);
  void resetCache();
};

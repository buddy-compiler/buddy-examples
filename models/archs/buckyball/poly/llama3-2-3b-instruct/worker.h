#pragma once
#include <buddy/Core/Container.h>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <thread>

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
  std::mutex mutex;
  std::condition_variable changed;
  int state = 0;
  bool ready = false, stopping = false;
  uint64_t operation, count, start, layer;
  char error[256];
};
struct Worker {
  Command *command;
  float *input, *output;
  int8_t *keyCodes, *keyScales, *valueCodes, *valueScales;
  std::thread thread;
};
void runWorker(Worker &, size_t rank, int cpu, int64_t *tokens,
               const std::filesystem::path &);

#pragma once
#include "generation.h"
#include <atomic>
#include <filesystem>
#include <sys/types.h>

int controlCpu(size_t tile);
class Execution {
public:
  struct Shared {
    std::atomic<int> state;
    size_t count;
    int64_t token, tokens[PrefillLength];
    float logits[MaxVocabSize];
    char error[256];
  };

private:
  Shared *shared;
  pid_t pid;
  int notify[2], completion[2];
  bool closed = false;
  void run(uint8_t operation);

public:
  Execution(const std::filesystem::path &directory,
            const std::vector<size_t> &tiles);
  ~Execution() noexcept(false);
  void prefill(const std::vector<int64_t> &tokens);
  void decode(int64_t token);
  const float *logits() const { return shared->logits; }
  void close();
};

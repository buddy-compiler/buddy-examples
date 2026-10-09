#pragma once
#include "execution.h"
#include "worker.h"
#include <vector>
struct Execution::State {
  Request request;
  void *storage;
  size_t storageBytes;
  bool closed = false, prefilled = false;
  size_t position = 0;
  int64_t *tokens;
  float *hidden;
  std::vector<Worker> workers;
  std::vector<float> reduced;
  State(const std::filesystem::path &, const std::vector<uint64_t> &);
  ~State() noexcept(false);
  void step(const std::vector<int64_t> &, size_t start);
  void close();
  void submit(size_t rank, uint64_t operation, size_t count, size_t start,
              size_t layer);
  void wait(size_t rank);
  void resetCache();
};

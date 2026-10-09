#pragma once

#include "task_api.h"
#include <cstdint>
#include <memory>
#include <thread>

class TaskWorker {
  struct Shared;
  std::unique_ptr<Shared> shared;
  std::thread worker;
  bool pending = false, closed = false;

public:
  TaskWorker(int cpu, void *workspace, size_t workspaceBytes);
  ~TaskWorker() noexcept(false);
  TaskWorker(const TaskWorker &) = delete;
  TaskWorker &operator=(const TaskWorker &) = delete;
  void waitReady();
  void submit(size_t kernel, const BufferView *inputs, size_t inputCount);
  void wait(BufferView *outputs, size_t outputCount);
  void release(void *allocation);
  void close();
};

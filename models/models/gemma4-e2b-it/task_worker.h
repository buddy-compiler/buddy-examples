#pragma once

#include "task_api.h"
#include <cstdint>
#include <sys/types.h>

class TaskWorker {
  struct Shared;
  Shared *shared;
  void *arena;
  size_t arenaBytes;
  pid_t pid;
  int request[2], completion[2];
  bool pending = false, closed = false;

public:
  TaskWorker(int cpu, void *workspace, size_t workspaceBytes);
  ~TaskWorker() noexcept(false);
  TaskWorker(const TaskWorker &) = delete;
  TaskWorker &operator=(const TaskWorker &) = delete;
  void submit(size_t kernel, const BufferView *inputs, size_t inputCount);
  void wait(BufferView *outputs, size_t outputCount);
  void release(void *allocation);
  void close();
};

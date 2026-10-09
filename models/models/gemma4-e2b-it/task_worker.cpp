#include "task_worker.h"
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <runtime.h>
#include <sched.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/wait.h>
#include <system_error>
#include <unistd.h>

struct TaskWorker::Shared {
  size_t kernel;
  BufferView inputs[GemmaTaskMaxInputs], outputs[GemmaTaskMaxOutputs];
  void *allocation;
};

TaskWorker::TaskWorker(int cpu, void *workspace, size_t workspaceBytes)
    : arena(workspace), arenaBytes(workspaceBytes) {
  shared = static_cast<Shared *>(mmap(nullptr, sizeof(Shared),
                                      PROT_READ | PROT_WRITE,
                                      MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  if (shared == MAP_FAILED || pipe(request) || pipe(completion))
    throw std::system_error(errno, std::generic_category(),
                            "create task worker");
  pid = fork();
  if (pid < 0)
    throw std::system_error(errno, std::generic_category(), "fork task worker");
  if (pid == 0) {
    ::close(request[1]);
    ::close(completion[0]);
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    CPU_SET(cpu, &cpus);
    if (sched_setaffinity(0, sizeof(cpus), &cpus))
      _exit(1);
    runtime_init(1024 * 1024);
    workspace_init(arena, arenaBytes);
    workspace_begin(arena, arenaBytes);
    uint8_t ready = 3;
    if (write(completion[1], &ready, 1) != 1)
      _exit(1);
    size_t next = 0;
    for (;;) {
      uint8_t operation;
      if (read(request[0], &operation, 1) != 1)
        _exit(1);
      if (operation == 2)
        std::exit(0);
      if (operation == 1) {
        workspace_free(shared->allocation);
      } else if (operation == 0) {
        size_t selected = 0;
        for (size_t step = 0; step < core_count(); ++step) {
          size_t core = (next + step) % core_count() + 1;
          if (core_signature(core) == GemmaTaskSignatures[shared->kernel]) {
            selected = core;
            break;
          }
        }
        if (!selected)
          _exit(1);
        next = selected % core_count();
        task_run_on(
            selected, GemmaTaskSignatures[shared->kernel],
            [](void *argument) {
              auto &job = *static_cast<Shared *>(argument);
              GemmaTaskEntries[job.kernel](job.inputs, job.outputs);
            },
            shared);
      } else {
        _exit(1);
      }
      if (write(completion[1], &operation, 1) != 1)
        _exit(1);
    }
  }
  ::close(request[0]);
  ::close(completion[1]);
  uint8_t ready;
  if (read(completion[0], &ready, 1) != 1 || ready != 3)
    throw std::runtime_error("tile task worker initialization failed");
}

void TaskWorker::submit(size_t kernel, const BufferView *inputs,
                        size_t inputCount) {
  if (pending || closed || kernel >= GemmaTaskEntryCount ||
      inputCount > GemmaTaskMaxInputs)
    throw std::runtime_error("invalid task submission");
  shared->kernel = kernel;
  std::memcpy(shared->inputs, inputs, inputCount * sizeof(BufferView));
  uint8_t operation = 0;
  if (write(request[1], &operation, 1) != 1)
    throw std::system_error(errno, std::generic_category(), "submit task");
  pending = true;
}

void TaskWorker::wait(BufferView *outputs, size_t outputCount) {
  if (!pending || outputCount > GemmaTaskMaxOutputs)
    throw std::runtime_error("invalid task wait");
  uint8_t operation;
  if (read(completion[0], &operation, 1) != 1 || operation != 0)
    throw std::runtime_error("tile task worker failed");
  std::memcpy(outputs, shared->outputs, outputCount * sizeof(BufferView));
  pending = false;
}

void TaskWorker::release(void *allocation) {
  if (pending || closed)
    throw std::runtime_error("cannot release an active task output");
  shared->allocation = allocation;
  uint8_t operation = 1;
  if (write(request[1], &operation, 1) != 1 ||
      read(completion[0], &operation, 1) != 1 || operation != 1)
    throw std::runtime_error("tile task release failed");
}

void TaskWorker::close() {
  if (pending || closed)
    throw std::runtime_error("cannot close an active or closed task worker");
  uint8_t operation = 2;
  if (write(request[1], &operation, 1) != 1)
    throw std::runtime_error("tile task shutdown failed");
  int status;
  if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status) ||
      WEXITSTATUS(status))
    throw std::runtime_error("tile task worker exited unsuccessfully");
  ::close(request[1]);
  ::close(completion[0]);
  munmap(shared, sizeof(Shared));
  closed = true;
}

TaskWorker::~TaskWorker() noexcept(false) {
  if (!closed)
    close();
}

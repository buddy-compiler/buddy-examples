#include "task_worker.h"
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <runtime.h>
#include <sched.h>
#include <stdexcept>
#include <system_error>

struct TaskWorker::Shared {
  std::mutex mutex;
  std::condition_variable changed;
  bool ready = false, command = false, complete = false;
  uint8_t operation;
  size_t kernel;
  BufferView inputs[GemmaTaskMaxInputs], outputs[GemmaTaskMaxOutputs];
  void *allocation;
};

TaskWorker::TaskWorker(int cpu, void *workspace, size_t workspaceBytes)
    : shared(std::make_unique<Shared>()) {
  worker = std::thread([this, cpu, workspace, workspaceBytes] {
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    CPU_SET(cpu, &cpus);
    if (sched_setaffinity(0, sizeof(cpus), &cpus))
      throw std::system_error(errno, std::generic_category(),
                              "pin tile worker");
    runtime_init(1024 * 1024);
    workspace_begin(workspace, workspaceBytes);
    std::unique_lock lock(shared->mutex);
    shared->ready = true;
    shared->changed.notify_one();
    size_t next = 0;
    for (;;) {
      shared->changed.wait(lock, [this] { return shared->command; });
      uint8_t operation = shared->operation;
      shared->command = false;
      lock.unlock();
      if (operation == 2) {
        runtime_shutdown();
        return;
      }
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
          throw std::runtime_error("no matching tile task core");
        next = selected % core_count();
        task_run_on(
            selected, GemmaTaskSignatures[shared->kernel],
            [](void *argument) {
              auto &job = *static_cast<Shared *>(argument);
              GemmaTaskEntries[job.kernel](job.inputs, job.outputs);
            },
            shared.get());
      } else {
        throw std::runtime_error("unknown tile worker operation");
      }
      lock.lock();
      shared->complete = true;
      shared->changed.notify_one();
    }
  });
}

void TaskWorker::waitReady() {
  std::unique_lock lock(shared->mutex);
  shared->changed.wait(lock, [this] { return shared->ready; });
}

void TaskWorker::submit(size_t kernel, const BufferView *inputs,
                        size_t inputCount) {
  if (pending || closed || kernel >= GemmaTaskEntryCount ||
      inputCount > GemmaTaskMaxInputs)
    throw std::runtime_error("invalid task submission");
  std::lock_guard lock(shared->mutex);
  shared->kernel = kernel;
  std::memcpy(shared->inputs, inputs, inputCount * sizeof(BufferView));
  shared->operation = 0;
  shared->complete = false;
  shared->command = true;
  pending = true;
  shared->changed.notify_one();
}

void TaskWorker::wait(BufferView *outputs, size_t outputCount) {
  if (!pending || outputCount > GemmaTaskMaxOutputs)
    throw std::runtime_error("invalid task wait");
  std::unique_lock lock(shared->mutex);
  shared->changed.wait(lock, [this] { return shared->complete; });
  std::memcpy(outputs, shared->outputs, outputCount * sizeof(BufferView));
  pending = false;
}

void TaskWorker::release(void *allocation) {
  if (pending || closed)
    throw std::runtime_error("cannot release an active task output");
  std::unique_lock lock(shared->mutex);
  shared->allocation = allocation;
  shared->operation = 1;
  shared->complete = false;
  shared->command = true;
  shared->changed.notify_one();
  shared->changed.wait(lock, [this] { return shared->complete; });
}

void TaskWorker::close() {
  if (pending || closed)
    throw std::runtime_error("cannot close an active or closed task worker");
  {
    std::lock_guard lock(shared->mutex);
    shared->operation = 2;
    shared->command = true;
    shared->changed.notify_one();
  }
  worker.join();
  closed = true;
}

TaskWorker::~TaskWorker() noexcept(false) {
  if (!closed)
    close();
}

#include "baremetal.h"

void runtime_baremetal_prepare_wait(uint32_t hartid) {
  asm volatile("csrsi mie, 8" ::: "memory");
  ((volatile uint32_t *)(uintptr_t)RUNTIME_CLINT_BASE)[hartid] = 0;
  asm volatile("fence iorw, iorw" ::: "memory");
}

void runtime_baremetal_wake(uint32_t hartid) {
  asm volatile("fence rw, ow" ::: "memory");
  ((volatile uint32_t *)(uintptr_t)RUNTIME_CLINT_BASE)[hartid] = 1;
}

void runtime_baremetal_notify(const struct runtime *runtime, uint64_t harts) {
  asm volatile("fence rw, ow" ::: "memory");
  for (size_t hart = 0; hart < runtime->hart_count; ++hart)
    if (harts >> hart & 1)
      ((volatile uint32_t *)(uintptr_t)
           RUNTIME_CLINT_BASE)[runtime->harts[hart].id] = 1;
}

struct worker {
  struct runtime *runtime;
  size_t hart;
};

static int work(void *context, void *argument) {
  struct worker *worker = argument;
  struct runtime *runtime = worker->runtime;
  for (;;) {
    runtime_baremetal_prepare_wait(runtime->harts[worker->hart].id);
    size_t job;
    const enum runtime_progress progress =
        runtime_claim(runtime, worker->hart, &job);
    if (progress == RUNTIME_DONE)
      return 0;
    if (progress == RUNTIME_CANCEL) {
      runtime_baremetal_notify(runtime, ~(uint64_t)0);
    } else if (progress == RUNTIME_WORK) {
      const struct runtime_job *request = &runtime->jobs[job];
      const int status = request->entry(context, request->argument);
      runtime_baremetal_notify(
          runtime, runtime_complete(runtime, worker->hart, job, status));
    } else {
      asm volatile("wfi" ::: "memory");
    }
  }
}

void runtime_baremetal_run(struct runtime *runtime, size_t hart) {
  struct worker worker = {runtime, hart};
  const struct runtime_hart *binding = &runtime->harts[hart];
  if (binding->execute(binding->context, work, &worker))
    __builtin_trap();
}

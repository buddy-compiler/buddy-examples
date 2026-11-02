#include "runtime.h"

struct worker {
  struct runtime *runtime;
  size_t core;
};

static int work(void *context, void *argument) {
  struct worker *worker = argument;
  struct runtime *runtime = worker->runtime;
  for (;;) {
    size_t job;
    enum runtime_progress progress = runtime_claim(runtime, worker->core, &job);
    if (progress == RUNTIME_DONE)
      return 0;
    if (progress == RUNTIME_WORK) {
      const struct runtime_job *request = &runtime->jobs[job];
      int status = request->entry(context, request->argument);
      runtime_complete(runtime, worker->core, job, status);
    }
  }
}

void runtime_baremetal_run(struct runtime *runtime, size_t core) {
  struct worker worker = {runtime, core};
  const struct runtime_core *binding = &runtime->cores[core];
  if (binding->execute(binding->context, work, &worker))
    __builtin_trap();
}

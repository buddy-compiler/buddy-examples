#include "runtime.h"

extern const struct runtime_binding __start_runtime_bindings[],
    __stop_runtime_bindings[];

struct runtime_core runtime_core(uint32_t id, uint64_t signature,
                                 void *context) {
  for (const struct runtime_binding *binding = __start_runtime_bindings;
       binding < __stop_runtime_bindings; ++binding)
    if (binding->signature == signature)
      return (struct runtime_core){.id = id,
                                   .signature = signature,
                                   .context = context,
                                   .execute = binding->execute};
  __builtin_trap();
}

void runtime_init(struct runtime *runtime, const struct runtime_core *cores,
                  size_t core_count, const struct runtime_job *jobs,
                  size_t job_count, struct runtime_result *results) {
  if (!core_count)
    __builtin_trap();
  for (size_t core = 0; core < core_count; ++core)
    for (size_t other = 0; other < core; ++other)
      if (cores[core].id == cores[other].id)
        __builtin_trap();
  for (size_t job = 0; job < job_count; ++job) {
    int compatible = 0;
    for (size_t core = 0; core < core_count; ++core)
      compatible |= (jobs[job].core == RUNTIME_ANY || jobs[job].core == core) &&
                    jobs[job].signature == cores[core].signature;
    if (!compatible || !jobs[job].entry)
      __builtin_trap();
    for (size_t dependency = 0; dependency < jobs[job].dependency_count;
         ++dependency)
      if (jobs[job].dependencies[dependency] >= job)
        __builtin_trap();
    results[job] = (struct runtime_result){RUNTIME_WAITING, 0, RUNTIME_ANY};
  }
  *runtime = (struct runtime){.cores = cores,
                              .core_count = core_count,
                              .jobs = jobs,
                              .job_count = job_count,
                              .results = results,
                              .remaining = job_count,
                              .lock = 0};
}

enum runtime_progress runtime_claim(struct runtime *runtime, size_t core,
                                    size_t *job) {
  if (core >= runtime->core_count)
    __builtin_trap();
  while (__atomic_exchange_n(&runtime->lock, 1, __ATOMIC_ACQUIRE)) {
  }
  for (size_t active = 0; active < runtime->job_count; ++active)
    if (runtime->results[active].state == RUNTIME_RUNNING &&
        runtime->results[active].core == core) {
      __atomic_store_n(&runtime->lock, 0, __ATOMIC_RELEASE);
      return RUNTIME_IDLE;
    }
  for (size_t candidate = 0; candidate < runtime->job_count; ++candidate) {
    struct runtime_result *result = &runtime->results[candidate];
    const struct runtime_job *request = &runtime->jobs[candidate];
    if (result->state != RUNTIME_WAITING)
      continue;
    int ready = 1, failed = 0;
    for (size_t dependency = 0; dependency < request->dependency_count;
         ++dependency) {
      const struct runtime_result *prior =
          &runtime->results[request->dependencies[dependency]];
      ready &= prior->state == RUNTIME_SUCCEEDED;
      failed |=
          prior->state == RUNTIME_FAILED || prior->state == RUNTIME_CANCELLED;
    }
    if (failed) {
      result->state = RUNTIME_CANCELLED;
      --runtime->remaining;
      continue;
    }
    if (!ready || (request->core != RUNTIME_ANY && request->core != core) ||
        request->signature != runtime->cores[core].signature)
      continue;
    result->state = RUNTIME_RUNNING;
    result->core = core;
    *job = candidate;
    __atomic_store_n(&runtime->lock, 0, __ATOMIC_RELEASE);
    return RUNTIME_WORK;
  }
  const enum runtime_progress progress =
      runtime->remaining ? RUNTIME_IDLE : RUNTIME_DONE;
  __atomic_store_n(&runtime->lock, 0, __ATOMIC_RELEASE);
  return progress;
}

void runtime_complete(struct runtime *runtime, size_t core, size_t job,
                      int status) {
  while (__atomic_exchange_n(&runtime->lock, 1, __ATOMIC_ACQUIRE)) {
  }
  struct runtime_result *result = &runtime->results[job];
  if (result->state != RUNTIME_RUNNING || result->core != core)
    __builtin_trap();
  result->status = status;
  result->state = status ? RUNTIME_FAILED : RUNTIME_SUCCEEDED;
  --runtime->remaining;
  __atomic_store_n(&runtime->lock, 0, __ATOMIC_RELEASE);
}

enum runtime_progress runtime_step(struct runtime *runtime, size_t core) {
  size_t job;
  const enum runtime_progress progress = runtime_claim(runtime, core, &job);
  if (progress == RUNTIME_WORK) {
    const struct runtime_job *request = &runtime->jobs[job];
    const int status = runtime->cores[core].execute(
        runtime->cores[core].context, request->entry, request->argument);
    runtime_complete(runtime, core, job, status);
  }
  return progress;
}

void runtime_cancel(struct runtime *runtime) {
  while (__atomic_exchange_n(&runtime->lock, 1, __ATOMIC_ACQUIRE)) {
  }
  for (size_t job = 0; job < runtime->job_count; ++job)
    if (runtime->results[job].state == RUNTIME_WAITING) {
      runtime->results[job].state = RUNTIME_CANCELLED;
      --runtime->remaining;
    }
  __atomic_store_n(&runtime->lock, 0, __ATOMIC_RELEASE);
}

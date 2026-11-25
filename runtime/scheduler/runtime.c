#include "runtime.h"

extern const struct runtime_binding __start_runtime_bindings[],
    __stop_runtime_bindings[];

struct runtime_hart runtime_hart(uint32_t id, uint64_t signature,
                                 void *context) {
  for (const struct runtime_binding *binding = __start_runtime_bindings;
       binding < __stop_runtime_bindings; ++binding)
    if (binding->signature == signature)
      return (struct runtime_hart){.id = id,
                                   .signature = signature,
                                   .context = context,
                                   .execute = binding->execute};
  __builtin_trap();
}

void runtime_init(struct runtime *runtime, struct runtime_hart *harts,
                  size_t hart_count, struct runtime_job *jobs, size_t job_count,
                  struct runtime_result *results) {
  if (!hart_count || hart_count > 64)
    __builtin_trap();
  for (size_t hart = 0; hart < hart_count; ++hart) {
    harts[hart].next = 0;
    for (size_t other = 0; other < hart; ++other)
      if (harts[hart].id == harts[other].id)
        __builtin_trap();
  }
  for (size_t job = 0; job < job_count; ++job) {
    if (jobs[job].hart >= hart_count ||
        jobs[job].signature != harts[jobs[job].hart].signature ||
        !jobs[job].entry)
      __builtin_trap();
    results[job] = (struct runtime_result){RUNTIME_WAITING, 0, 0};
  }
  for (size_t job = 0; job < job_count; ++job)
    for (size_t dependency = 0; dependency < jobs[job].dependency_count;
         ++dependency) {
      if (jobs[job].dependencies[dependency] >= job)
        __builtin_trap();
      results[jobs[job].dependencies[dependency]].dependents |=
          (uint64_t)1 << jobs[job].hart;
    }
  *runtime = (struct runtime){.harts = harts,
                              .hart_count = hart_count,
                              .jobs = jobs,
                              .job_count = job_count,
                              .results = results,
                              .remaining = job_count,
                              .lock = 0};
}

size_t runtime_submit(struct runtime *runtime, const struct runtime_job *job) {
  while (__atomic_exchange_n(&runtime->lock, 1, __ATOMIC_ACQUIRE)) {
    while (__atomic_load_n(&runtime->lock, __ATOMIC_RELAXED)) {
    }
  }
  const size_t index = runtime->job_count;
  runtime->jobs[index] = *job;
  runtime->results[index] = (struct runtime_result){RUNTIME_WAITING, 0, 0};
  /* A dependency that completes after this sees the hart; one that completed
   * before is seen by the hart, which its submitter wakes. */
  for (size_t dependency = 0; dependency < job->dependency_count; ++dependency)
    __atomic_fetch_or(
        &runtime->results[job->dependencies[dependency]].dependents,
        (uint64_t)1 << job->hart, __ATOMIC_SEQ_CST);
  __atomic_add_fetch(&runtime->remaining, 1, __ATOMIC_RELEASE);
  __atomic_store_n(&runtime->job_count, index + 1, __ATOMIC_RELEASE);
  __atomic_store_n(&runtime->lock, 0, __ATOMIC_RELEASE);
  return index;
}

enum runtime_progress runtime_claim(struct runtime *runtime, size_t hart,
                                    size_t *job) {
  struct runtime_hart *self = &runtime->harts[hart];
  const size_t count = __atomic_load_n(&runtime->job_count, __ATOMIC_ACQUIRE);
  while (self->next < count && runtime->jobs[self->next].hart != hart)
    ++self->next;
  if (self->next == count)
    return __atomic_load_n(&runtime->remaining, __ATOMIC_ACQUIRE)
               ? RUNTIME_IDLE
               : RUNTIME_DONE;
  const struct runtime_job *request = &runtime->jobs[self->next];
  int ready = 1, failed = 0;
  for (size_t dependency = 0; dependency < request->dependency_count;
       ++dependency) {
    const enum runtime_state state = __atomic_load_n(
        &runtime->results[request->dependencies[dependency]].state,
        __ATOMIC_SEQ_CST);
    ready &= state == RUNTIME_SUCCEEDED;
    failed |= state == RUNTIME_FAILED || state == RUNTIME_CANCELLED;
  }
  struct runtime_result *result = &runtime->results[self->next];
  if (failed) {
    __atomic_store_n(&result->state, RUNTIME_CANCELLED, __ATOMIC_SEQ_CST);
    __atomic_sub_fetch(&runtime->remaining, 1, __ATOMIC_ACQ_REL);
    ++self->next;
    return RUNTIME_CANCEL;
  }
  if (!ready)
    return RUNTIME_IDLE;
  __atomic_store_n(&result->state, RUNTIME_RUNNING, __ATOMIC_RELEASE);
  *job = self->next++;
  return RUNTIME_WORK;
}

uint64_t runtime_complete(struct runtime *runtime, size_t hart, size_t job,
                          int status) {
  struct runtime_result *result = &runtime->results[job];
  if (result->state != RUNTIME_RUNNING || runtime->jobs[job].hart != hart)
    __builtin_trap();
  result->status = status;
  __atomic_store_n(&result->state, status ? RUNTIME_FAILED : RUNTIME_SUCCEEDED,
                   __ATOMIC_SEQ_CST);
  const uint64_t dependents =
      __atomic_load_n(&result->dependents, __ATOMIC_SEQ_CST);
  if (!__atomic_sub_fetch(&runtime->remaining, 1, __ATOMIC_ACQ_REL) || status)
    return ~(uint64_t)0;
  return dependents;
}

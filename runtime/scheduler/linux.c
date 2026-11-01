#define _GNU_SOURCE
#include "linux.h"
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>

static void check(int status) {
  if (status) {
    fprintf(stderr, "Runtime pthread operation failed: %d\n", status);
    abort();
  }
}

static void *run(void *argument) {
  struct runtime_linux_worker *worker = argument;
  struct runtime *runtime = worker->runtime;
  check(pthread_mutex_lock(worker->mutex));
  for (;;) {
    size_t job;
    const size_t remaining = runtime->remaining;
    const enum runtime_progress progress =
        runtime_claim(runtime, worker->core, &job);
    if (runtime->remaining != remaining)
      check(pthread_cond_broadcast(worker->changed));
    if (progress == RUNTIME_DONE) {
      check(pthread_cond_broadcast(worker->changed));
      break;
    }
    if (progress == RUNTIME_IDLE) {
      check(pthread_cond_wait(worker->changed, worker->mutex));
      continue;
    }
    check(pthread_mutex_unlock(worker->mutex));
    const struct runtime_job *request = &runtime->jobs[job];
    const struct runtime_core *core = &runtime->cores[worker->core];
    const int status =
        core->execute(core->context, request->entry, request->argument);
    check(pthread_mutex_lock(worker->mutex));
    runtime_complete(runtime, worker->core, job, status);
    check(pthread_cond_broadcast(worker->changed));
  }
  check(pthread_mutex_unlock(worker->mutex));
  return NULL;
}

size_t runtime_linux_submit(struct runtime_linux_worker *worker,
                            const struct runtime_job *job) {
  check(pthread_mutex_lock(worker->mutex));
  const size_t index = runtime_submit(worker->runtime, job);
  check(pthread_cond_broadcast(worker->changed));
  check(pthread_mutex_unlock(worker->mutex));
  return index;
}

int runtime_linux_wait(struct runtime_linux_worker *worker, size_t job) {
  struct runtime *runtime = worker->runtime;
  if (runtime->jobs[job].core == worker->core)
    abort();
  check(pthread_mutex_lock(worker->mutex));
  while (runtime->results[job].state == RUNTIME_WAITING ||
         runtime->results[job].state == RUNTIME_RUNNING)
    check(pthread_cond_wait(worker->changed, worker->mutex));
  const struct runtime_result result = runtime->results[job];
  check(pthread_mutex_unlock(worker->mutex));
  if (result.state == RUNTIME_CANCELLED)
    return -1;
  return result.status;
}

void runtime_linux_run(struct runtime *runtime,
                       struct runtime_linux_worker *workers,
                       const unsigned *cpus, size_t stack_bytes) {
  pthread_mutex_t mutex;
  pthread_cond_t changed;
  pthread_attr_t attributes;
  for (size_t core = 0; core < runtime->core_count; ++core) {
    const unsigned cpu = cpus[core];
    for (size_t other = 0; other < core; ++other)
      if (cpu == cpus[other]) {
        fputs("Runtime cores must use distinct Linux CPUs\n", stderr);
        abort();
      }
  }
  check(pthread_mutex_init(&mutex, NULL));
  check(pthread_cond_init(&changed, NULL));
  check(pthread_attr_init(&attributes));
  check(pthread_attr_setstacksize(&attributes, stack_bytes));
  for (size_t core = 0; core < runtime->core_count; ++core) {
    workers[core].runtime = runtime;
    workers[core].core = core;
    workers[core].mutex = &mutex;
    workers[core].changed = &changed;
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(cpus[core], &affinity);
    check(
        pthread_attr_setaffinity_np(&attributes, sizeof(affinity), &affinity));
    check(pthread_create(&workers[core].thread, &attributes, run,
                         &workers[core]));
  }
  check(pthread_attr_destroy(&attributes));
  for (size_t core = 0; core < runtime->core_count; ++core)
    check(pthread_join(workers[core].thread, NULL));
  check(pthread_cond_destroy(&changed));
  check(pthread_mutex_destroy(&mutex));
}

int runtime_linux_job(unsigned cpu, uint64_t signature,
                      int (*entry)(void *, void *), void *argument) {
  struct runtime_core core = runtime_core(cpu, signature, NULL);
  struct runtime_job job = {
      .signature = signature, .core = 0, .entry = entry, .argument = argument};
  struct runtime_result result;
  struct runtime runtime;
  struct runtime_linux_worker worker;
  runtime_init(&runtime, &core, 1, &job, 1, &result);
  // Model jobs run deep compiled call chains; the stack is virtual until
  // touched.
  runtime_linux_run(&runtime, &worker, &cpu, (size_t)64 << 20);
  if (result.state != RUNTIME_SUCCEEDED && result.state != RUNTIME_FAILED)
    abort();
  return result.status;
}

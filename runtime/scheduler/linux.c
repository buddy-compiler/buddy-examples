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

/* Called with the run's mutex held. Wake only a ready worker or a completed
 * result's waiter; cancellation and shutdown wake all sleeping workers. */
static void notify(struct runtime_linux_worker *worker, int cancelled) {
  struct runtime *runtime = worker->runtime;
  struct runtime_linux_worker *workers = worker->workers;
  int wake[runtime->core_count];
  for (size_t core = 0; core < runtime->core_count; ++core)
    wake[core] =
        workers[core].sleeping &&
        (cancelled || !__atomic_load_n(&runtime->remaining, __ATOMIC_ACQUIRE) ||
         (workers[core].waiting_for != RUNTIME_ANY &&
          __atomic_load_n(&runtime->results[workers[core].waiting_for].state,
                          __ATOMIC_ACQUIRE) >= RUNTIME_SUCCEEDED));
  for (size_t job = runtime->first; job < runtime->job_count; ++job) {
    if (__atomic_load_n(&runtime->results[job].state, __ATOMIC_ACQUIRE) !=
        RUNTIME_WAITING)
      continue;
    const struct runtime_job *request = &runtime->jobs[job];
    int ready = 1;
    for (size_t dependency = 0; dependency < request->dependency_count;
         ++dependency)
      ready &= __atomic_load_n(
                   &runtime->results[request->dependencies[dependency]].state,
                   __ATOMIC_ACQUIRE) == RUNTIME_SUCCEEDED;
    if (!ready)
      continue;
    for (size_t core = 0; core < runtime->core_count; ++core)
      if (!wake[core] && workers[core].sleeping &&
          workers[core].waiting_for == RUNTIME_ANY &&
          (request->core == RUNTIME_ANY || request->core == core) &&
          request->signature == runtime->cores[core].signature) {
        wake[core] = 1;
        break;
      }
  }
  for (size_t core = 0; core < runtime->core_count; ++core)
    if (wake[core])
      check(pthread_cond_signal(&workers[core].changed));
}

static int work(void *context, void *argument) {
  struct runtime_linux_worker *worker = argument;
  struct runtime *runtime = worker->runtime;
  check(pthread_mutex_lock(worker->mutex));
  for (;;) {
    size_t job;
    const size_t remaining = runtime->remaining;
    const enum runtime_progress progress =
        runtime_claim(runtime, worker->core, &job);
    if (runtime->remaining != remaining)
      notify(worker, 1);
    if (progress == RUNTIME_DONE) {
      notify(worker, 0);
      break;
    }
    if (progress == RUNTIME_IDLE) {
      worker->sleeping = 1;
      worker->waiting_for = RUNTIME_ANY;
      check(pthread_cond_wait(&worker->changed, worker->mutex));
      worker->sleeping = 0;
      continue;
    }
    check(pthread_mutex_unlock(worker->mutex));
    const struct runtime_job *request = &runtime->jobs[job];
    const int status = request->entry(context, request->argument);
    check(pthread_mutex_lock(worker->mutex));
    runtime_complete(runtime, worker->core, job, status);
    notify(worker, status != 0);
  }
  check(pthread_mutex_unlock(worker->mutex));
  return 0;
}

static void *run(void *argument) {
  struct runtime_linux_worker *worker = argument;
  const struct runtime_core *core = &worker->runtime->cores[worker->core];
  if (core->execute(core->context, work, worker))
    abort();
  return NULL;
}

size_t runtime_linux_submit(struct runtime_linux_worker *worker,
                            const struct runtime_job *job) {
  check(pthread_mutex_lock(worker->mutex));
  const size_t index = runtime_submit(worker->runtime, job);
  notify(worker, 0);
  check(pthread_mutex_unlock(worker->mutex));
  return index;
}

int runtime_linux_wait(struct runtime_linux_worker *worker, size_t job) {
  struct runtime *runtime = worker->runtime;
  if (runtime->jobs[job].core == worker->core)
    abort();
  check(pthread_mutex_lock(worker->mutex));
  while (runtime->results[job].state == RUNTIME_WAITING ||
         runtime->results[job].state == RUNTIME_RUNNING) {
    worker->sleeping = 1;
    worker->waiting_for = job;
    check(pthread_cond_wait(&worker->changed, worker->mutex));
    worker->sleeping = 0;
  }
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
  check(pthread_attr_init(&attributes));
  check(pthread_attr_setstacksize(&attributes, stack_bytes));
  for (size_t core = 0; core < runtime->core_count; ++core) {
    workers[core].runtime = runtime;
    workers[core].core = core;
    workers[core].mutex = &mutex;
    workers[core].workers = workers;
    workers[core].sleeping = 0;
    workers[core].waiting_for = RUNTIME_ANY;
    check(pthread_cond_init(&workers[core].changed, NULL));
  }
  for (size_t core = 0; core < runtime->core_count; ++core) {
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
  for (size_t core = 0; core < runtime->core_count; ++core)
    check(pthread_cond_destroy(&workers[core].changed));
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

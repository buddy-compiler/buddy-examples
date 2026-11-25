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

/* Called with the run's mutex held: wakes the sleeping workers of `harts`, a
 * bit per index. */
static void notify(struct runtime_linux_worker *worker, uint64_t harts) {
  struct runtime_linux_worker *workers = worker->workers;
  for (size_t hart = 0; hart < worker->runtime->hart_count; ++hart)
    if (harts >> hart & 1 && workers[hart].sleeping) {
      workers[hart].sleeping = 0;
      check(pthread_cond_signal(&workers[hart].changed));
    }
}

static int work(void *context, void *argument) {
  struct runtime_linux_worker *worker = argument;
  struct runtime *runtime = worker->runtime;
  struct runtime_linux *team = worker->team;
  uint64_t generation = 0;
  check(pthread_mutex_lock(worker->mutex));
  for (;;) {
    while (generation == team->generation && !team->stopping)
      check(pthread_cond_wait(&worker->changed, worker->mutex));
    if (team->stopping)
      break;
    generation = team->generation;
    for (;;) {
      size_t job;
      const enum runtime_progress progress =
          runtime_claim(runtime, worker->hart, &job);
      if (progress == RUNTIME_DONE)
        break;
      if (progress == RUNTIME_CANCEL) {
        notify(worker, ~(uint64_t)0);
        continue;
      }
      if (progress == RUNTIME_IDLE) {
        worker->sleeping = 1;
        check(pthread_cond_wait(&worker->changed, worker->mutex));
        worker->sleeping = 0;
        continue;
      }
      check(pthread_mutex_unlock(worker->mutex));
      const struct runtime_job *request = &runtime->jobs[job];
      const int status = request->entry(context, request->argument);
      check(pthread_mutex_lock(worker->mutex));
      notify(worker, runtime_complete(runtime, worker->hart, job, status));
    }
    if (++team->completed == runtime->hart_count)
      check(pthread_cond_signal(&team->drained));
  }
  check(pthread_mutex_unlock(worker->mutex));
  return 0;
}

static void *run(void *argument) {
  struct runtime_linux_worker *worker = argument;
  const struct runtime_hart *hart = &worker->runtime->harts[worker->hart];
  if (hart->execute(hart->context, work, worker))
    abort();
  return NULL;
}

size_t runtime_linux_submit(struct runtime_linux_worker *worker,
                            const struct runtime_job *job) {
  check(pthread_mutex_lock(worker->mutex));
  const size_t index = runtime_submit(worker->runtime, job);
  notify(worker, (uint64_t)1 << job->hart);
  check(pthread_mutex_unlock(worker->mutex));
  return index;
}

int runtime_linux_wait(struct runtime_linux_worker *worker, size_t job) {
  struct runtime *runtime = worker->runtime;
  if (runtime->jobs[job].hart == worker->hart)
    abort();
  check(pthread_mutex_lock(worker->mutex));
  runtime->results[job].dependents |= (uint64_t)1 << worker->hart;
  while (runtime->results[job].state == RUNTIME_WAITING ||
         runtime->results[job].state == RUNTIME_RUNNING) {
    worker->sleeping = 1;
    check(pthread_cond_wait(&worker->changed, worker->mutex));
    worker->sleeping = 0;
  }
  const struct runtime_result result = runtime->results[job];
  check(pthread_mutex_unlock(worker->mutex));
  if (result.state == RUNTIME_CANCELLED)
    return -1;
  return result.status;
}

void runtime_linux_init(struct runtime_linux *team, struct runtime *runtime,
                        struct runtime_linux_worker *workers,
                        const unsigned *cpus, size_t stack_bytes) {
  *team = (struct runtime_linux){
      .runtime = runtime, .workers = workers, .completed = runtime->hart_count};
  pthread_attr_t attributes;
  for (size_t hart = 0; hart < runtime->hart_count; ++hart) {
    const unsigned cpu = cpus[hart];
    for (size_t other = 0; other < hart; ++other)
      if (cpu == cpus[other]) {
        fputs("Runtime harts must use distinct Linux CPUs\n", stderr);
        abort();
      }
  }
  check(pthread_mutex_init(&team->mutex, NULL));
  check(pthread_cond_init(&team->drained, NULL));
  check(pthread_attr_init(&attributes));
  check(pthread_attr_setstacksize(&attributes, stack_bytes));
  for (size_t hart = 0; hart < runtime->hart_count; ++hart) {
    workers[hart].runtime = runtime;
    workers[hart].hart = hart;
    workers[hart].mutex = &team->mutex;
    workers[hart].team = team;
    workers[hart].workers = workers;
    workers[hart].sleeping = 0;
    check(pthread_cond_init(&workers[hart].changed, NULL));
  }
  for (size_t hart = 0; hart < runtime->hart_count; ++hart) {
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(cpus[hart], &affinity);
    check(
        pthread_attr_setaffinity_np(&attributes, sizeof(affinity), &affinity));
    check(pthread_create(&workers[hart].thread, &attributes, run,
                         &workers[hart]));
  }
  check(pthread_attr_destroy(&attributes));
}

void runtime_linux_run(struct runtime_linux *team) {
  check(pthread_mutex_lock(&team->mutex));
  team->completed = 0;
  ++team->generation;
  for (size_t hart = 0; hart < team->runtime->hart_count; ++hart)
    check(pthread_cond_signal(&team->workers[hart].changed));
  while (team->completed != team->runtime->hart_count)
    check(pthread_cond_wait(&team->drained, &team->mutex));
  check(pthread_mutex_unlock(&team->mutex));
}

void runtime_linux_destroy(struct runtime_linux *team) {
  struct runtime *runtime = team->runtime;
  struct runtime_linux_worker *workers = team->workers;
  check(pthread_mutex_lock(&team->mutex));
  team->stopping = 1;
  for (size_t hart = 0; hart < runtime->hart_count; ++hart)
    check(pthread_cond_signal(&workers[hart].changed));
  check(pthread_mutex_unlock(&team->mutex));
  for (size_t hart = 0; hart < runtime->hart_count; ++hart)
    check(pthread_join(workers[hart].thread, NULL));
  for (size_t hart = 0; hart < runtime->hart_count; ++hart)
    check(pthread_cond_destroy(&workers[hart].changed));
  check(pthread_cond_destroy(&team->drained));
  check(pthread_mutex_destroy(&team->mutex));
}

int runtime_linux_job(unsigned cpu, uint64_t signature,
                      int (*entry)(void *, void *), void *argument) {
  struct runtime_hart hart = runtime_hart(cpu, signature, NULL);
  struct runtime_job job = {
      .signature = signature, .hart = 0, .entry = entry, .argument = argument};
  struct runtime_result result;
  struct runtime runtime;
  struct runtime_linux_worker worker;
  struct runtime_linux team;
  runtime_init(&runtime, &hart, 1, &job, 1, &result);
  // Model jobs run deep compiled call chains; the stack is virtual until
  // touched.
  runtime_linux_init(&team, &runtime, &worker, &cpu, (size_t)64 << 20);
  runtime_linux_run(&team);
  runtime_linux_destroy(&team);
  if (result.state != RUNTIME_SUCCEEDED && result.state != RUNTIME_FAILED)
    abort();
  return result.status;
}

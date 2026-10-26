#pragma once

#include "runtime.h"
#include <pthread.h>

struct runtime_linux_worker {
  pthread_t thread;
  struct runtime *runtime;
  size_t core;
  pthread_mutex_t *mutex;
  pthread_cond_t *changed;
};

#ifdef __cplusplus
extern "C" {
#endif

/* cpus[core] is the explicit Linux CPU affinity for that core. The caller
 * provides core_count worker slots. Return after all workers are joined. */
void runtime_linux_run(struct runtime *runtime,
                       struct runtime_linux_worker *workers,
                       const unsigned *cpus, size_t stack_bytes);

/* Runs one job on the core at Linux CPU `cpu` (its platform identity) and
 * returns the job's status; jobs of a core with threads receive its engine. */
int runtime_linux_job(unsigned cpu, uint64_t signature,
                      int (*entry)(void *, void *), void *argument);

#ifdef __cplusplus
}
#endif

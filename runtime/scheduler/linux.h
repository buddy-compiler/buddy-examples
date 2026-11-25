#pragma once

#include "runtime.h"
#include <pthread.h>

struct runtime_linux;

struct runtime_linux_worker {
  pthread_t thread;
  struct runtime *runtime;
  size_t hart;
  pthread_mutex_t *mutex;
  pthread_cond_t changed;
  struct runtime_linux_worker *workers;
  int sleeping;
  struct runtime_linux *team;
};

struct runtime_linux {
  struct runtime *runtime;
  struct runtime_linux_worker *workers;
  pthread_mutex_t mutex;
  pthread_cond_t drained;
  size_t completed;
  uint64_t generation;
  int stopping;
};

#ifdef __cplusplus
extern "C" {
#endif

/* Pinned workers retain their hart execution contexts across batches.
 * Reinitialize the runtime's jobs only between calls to runtime_linux_run;
 * the harts and worker storage remain alive until runtime_linux_destroy. */
void runtime_linux_init(struct runtime_linux *team, struct runtime *runtime,
                        struct runtime_linux_worker *workers,
                        const unsigned *cpus, size_t stack_bytes);
void runtime_linux_run(struct runtime_linux *team);
void runtime_linux_destroy(struct runtime_linux *team);

/* A running job can submit work to another hart and wait for it. The caller
 * uses its own worker; waiting on its own hart would deadlock. */
size_t runtime_linux_submit(struct runtime_linux_worker *worker,
                            const struct runtime_job *job);
int runtime_linux_wait(struct runtime_linux_worker *worker, size_t job);

/* Runs one job on the hart at Linux CPU `cpu` (its platform identity) and
 * returns the job's status; jobs of a hart with threads receive its kernel. */
int runtime_linux_job(unsigned cpu, uint64_t signature,
                      int (*entry)(void *, void *), void *argument);

#ifdef __cplusplus
}
#endif

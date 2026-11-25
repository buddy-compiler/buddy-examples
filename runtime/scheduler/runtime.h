#pragma once

#include <stddef.h>
#include <stdint.h>

enum runtime_state {
  RUNTIME_WAITING,
  RUNTIME_RUNNING,
  RUNTIME_SUCCEEDED,
  RUNTIME_FAILED,
  RUNTIME_CANCELLED
};

/* RUNTIME_CANCEL: the claim cancelled the next job of the hart, whose
 * dependency failed or was cancelled; the worker wakes every hart and claims
 * again. */
enum runtime_progress {
  RUNTIME_IDLE,
  RUNTIME_WORK,
  RUNTIME_DONE,
  RUNTIME_CANCEL
};

struct runtime_hart {
  /* Platform identity, independent of an OS CPU affinity number. */
  uint32_t id;
  /* Hart execution profile accepted by this worker. */
  uint64_t signature;
  void *context;
  int (*execute)(void *context, int (*entry)(void *, void *), void *argument);
  /* The first job of this hart not claimed yet; its worker's alone. */
  size_t next;
};

/* Each kind of hart registers how it executes an entry in the
 * `runtime_bindings` section: static const struct runtime_binding binding
 *       __attribute__((section("runtime_bindings"), used)) = {HART_SIGNATURE,
 * execute}; A hart without hardware threads calls the entry directly; a hart
 * with threads passes its kernel as the entry's hart context. A worker runs
 * its entire batch inside this entry, keeping the context alive across jobs. */
struct runtime_binding {
  uint64_t signature;
  int (*execute)(void *context, int (*entry)(void *, void *), void *argument);
};

/* The jobs of a hart run one at a time, in the order of their indices: a job
 * starts once the jobs of its hart before it ended and its dependencies
 * succeeded. */
struct runtime_job {
  uint64_t signature;
  /* Index in the runtime's hart array. */
  size_t hart;
  /* Finish this hart job before returning: zero succeeds, nonzero fails. */
  int (*entry)(void *hart_context, void *argument);
  void *argument;
  /* Dependency indices refer to earlier jobs in this batch. */
  const size_t *dependencies;
  size_t dependency_count;
};

struct runtime_result {
  enum runtime_state state;
  /* Entry return code for SUCCEEDED/FAILED; inspect state for cancellation. */
  int status;
  /* The harts, a bit per index, that wait for this job: the harts of the jobs
   * that depend on it and the waiters of runtime_linux_wait or of the device
   * program. Its completion wakes them. */
  uint64_t dependents;
};

/* The caller owns all arrays until every worker has returned. Read results
 * after joining the workers; callbacks may read successful dependencies. */
struct runtime {
  struct runtime_hart *harts;
  size_t hart_count;
  struct runtime_job *jobs;
  size_t job_count;
  struct runtime_result *results;
  size_t remaining;
  /* Orders the submissions. */
  uint32_t lock;
};

#ifdef __cplusplus
extern "C" {
#endif

/* The hart with platform identity `id`, executing jobs the way its signature
 * registered. */
struct runtime_hart runtime_hart(uint32_t id, uint64_t signature,
                                 void *context);

/* At most 64 harts: a hart is a bit of runtime_result.dependents. */
void runtime_init(struct runtime *runtime, struct runtime_hart *harts,
                  size_t hart_count, struct runtime_job *jobs, size_t job_count,
                  struct runtime_result *results);

/* Appends a job to the batch while the workers run it; returns its index.
 * The caller's arrays hold it. A batch stays open while one of its jobs runs:
 * a job may submit the jobs that follow it. The caller wakes the job's hart.
 */
size_t runtime_submit(struct runtime *runtime, const struct runtime_job *job);

/* Each hart has exactly one worker; only it claims the jobs of its hart and
 * completes them. A claim looks at the next job of the hart only. */
enum runtime_progress runtime_claim(struct runtime *runtime, size_t hart,
                                    size_t *job);
/* Returns the harts to wake: those waiting for the job, every hart if it
 * failed or was the last. */
uint64_t runtime_complete(struct runtime *runtime, size_t hart, size_t job,
                          int status);

/* Every participating hart calls this with its explicit hart index. Hart
 * startup, clocks and cross-hart memory coherence belong to the platform. */
void runtime_baremetal_run(struct runtime *runtime, size_t hart);

#ifdef __cplusplus
}
#endif

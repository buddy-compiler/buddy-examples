#pragma once

#include <stddef.h>
#include <stdint.h>

#define RUNTIME_ANY SIZE_MAX

enum runtime_state {
  RUNTIME_WAITING,
  RUNTIME_RUNNING,
  RUNTIME_SUCCEEDED,
  RUNTIME_FAILED,
  RUNTIME_CANCELLED
};

enum runtime_progress { RUNTIME_IDLE, RUNTIME_WORK, RUNTIME_DONE };

struct runtime_core {
  /* Platform identity, independent of an OS CPU affinity number. */
  uint32_t id;
  /* Core execution profile accepted by this worker. */
  uint64_t signature;
  void *context;
  int (*execute)(void *context, int (*entry)(void *, void *), void *argument);
};

/* Each kind of core registers how it executes a job in the `runtime_bindings`
 * section: static const struct runtime_binding binding
 *       __attribute__((section("runtime_bindings"), used)) = {CORE_SIGNATURE,
 * execute}; A core without hardware threads calls the entry directly; a core
 * with threads passes its engine as the entry's core context. */
struct runtime_binding {
  uint64_t signature;
  int (*execute)(void *context, int (*entry)(void *, void *), void *argument);
};

struct runtime_job {
  uint64_t signature;
  /* Index in the runtime's core array, or RUNTIME_ANY. */
  size_t core;
  /* Finish this core job before returning: zero succeeds, nonzero fails. */
  int (*entry)(void *core_context, void *argument);
  void *argument;
  /* Dependency indices refer to earlier jobs in this batch. */
  const size_t *dependencies;
  size_t dependency_count;
};

struct runtime_result {
  enum runtime_state state;
  /* Entry return code for SUCCEEDED/FAILED; inspect state for cancellation. */
  int status;
  size_t core;
};

/* The caller owns all arrays until every worker has returned. Read results
 * after joining the workers; callbacks may read successful dependencies. */
struct runtime {
  const struct runtime_core *cores;
  size_t core_count;
  const struct runtime_job *jobs;
  size_t job_count;
  struct runtime_result *results;
  size_t remaining;
  uint32_t lock;
};

#ifdef __cplusplus
extern "C" {
#endif

/* The core with platform identity `id`, executing jobs the way its signature
 * registered. */
struct runtime_core runtime_core(uint32_t id, uint64_t signature,
                                 void *context);

void runtime_init(struct runtime *runtime, const struct runtime_core *cores,
                  size_t core_count, const struct runtime_job *jobs,
                  size_t job_count, struct runtime_result *results);

/* Each core has exactly one worker. A claim must be completed by that worker.
 */
enum runtime_progress runtime_claim(struct runtime *runtime, size_t core,
                                    size_t *job);
void runtime_complete(struct runtime *runtime, size_t core, size_t job,
                      int status);
enum runtime_progress runtime_step(struct runtime *runtime, size_t core);

/* Running callbacks drain; jobs that have not started are cancelled. */
void runtime_cancel(struct runtime *runtime);

/* Every participating hart calls this with its explicit core index. Hart
 * startup, clocks and cross-core memory coherence belong to the platform. */
void runtime_baremetal_run(struct runtime *runtime, size_t core);

#ifdef __cplusplus
}
#endif

#pragma once
#include "thread.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ENGINE_ANY_THREAD UINT32_MAX

enum engine_state {
  ENGINE_QUEUED,
  ENGINE_RUNNING,
  ENGINE_CANCELLING,
  ENGINE_SUCCEEDED,
  ENGINE_CANCELLED
};

/* Each thread program registers one binding in the `engine_bindings` section:
 *   static const struct engine_binding binding
 *       __attribute__((section("engine_bindings"), used)) = {THREAD_SIGNATURE,
 * prepare};
 */
struct engine_binding {
  uint64_t signature;
  void (*prepare)(uint32_t thread, void *opaque, struct thread_task *task);
};

struct engine_slot {
  uint64_t signature, code_bytes, tls_base, tls_bytes, tss_base, tss_bytes;
  int online;
  size_t job;
  const struct engine_binding *binding;
};

struct engine_job {
  uint32_t thread;
  struct thread_task task;
  const size_t *dependencies;
  size_t dependency_count;
  void *opaque;
};

struct engine_result {
  enum engine_state state;
  uint32_t thread;
  uint64_t value;
};

/* One software owner on one OS hart. All arrays remain live until engine_end.
 */
struct engine {
  struct engine_slot *slots;
  size_t thread_count;
  const struct engine_job *jobs;
  struct engine_result *results;
  size_t job_count, completed;
  int active;
};

/* Also makes `engine` the calling host thread's engine_current(). */
void engine_init(struct engine *engine, struct engine_slot *slots,
                 size_t capacity);
struct engine *engine_current(void);
/* Dependencies name earlier jobs: every dependency index must be less than its
 * job. */
void engine_begin(struct engine *engine, const struct engine_job *jobs,
                  size_t count, struct engine_result *results);
/* Poll every running thread, acknowledge completions, then launch ready jobs.
 */
size_t engine_poll(struct engine *engine);
void engine_wait(struct engine *engine);
void engine_cancel(struct engine *engine, size_t job);
void engine_cancel_all(struct engine *engine);
/* Release only after all tasks, including cancelled work, have completed. */
void engine_end(struct engine *engine);
size_t engine_run(struct engine *engine, const struct engine_job *jobs,
                  size_t count, struct engine_result *results);

#ifdef __cplusplus
}
#endif

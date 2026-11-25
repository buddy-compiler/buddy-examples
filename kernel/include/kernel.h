#pragma once
#include "thread.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KERNEL_ANY_THREAD UINT32_MAX

enum kernel_state {
  KERNEL_QUEUED,
  KERNEL_RUNNING,
  KERNEL_CANCELLING,
  KERNEL_SUCCEEDED,
  KERNEL_CANCELLED
};

/* Each thread program registers one binding in the `kernel_bindings` section:
 *   static const struct kernel_binding binding
 *       __attribute__((section("kernel_bindings"), used)) = {THREAD_SIGNATURE,
 * load, launch};
 * kernel_init calls `load` once per thread of the signature, to load the
 * program; `launch` loads a job's arguments and starts its task. */
struct kernel_binding {
  uint64_t signature;
  void (*load)(uint32_t thread);
  void (*launch)(uint32_t thread, void *opaque, struct thread_task *task);
};

struct kernel_slot {
  uint64_t signature, code_bytes, tls_base, tls_bytes, tss_base, tss_bytes;
  int online, loaded;
  size_t job;
  const struct kernel_binding *binding;
};

struct kernel_job {
  uint32_t thread;
  struct thread_task task;
  const size_t *dependencies;
  size_t dependency_count;
  void *opaque;
  /* No data accesses; dependencies and cancellation still apply. */
  int empty;
};

struct kernel_result {
  enum kernel_state state;
  uint32_t thread;
  uint64_t value;
};

/* A compiled edge between two jobs. The kernel uses UINT64_MAX when no
 * physical transfer is needed. */
struct kernel_transfer {
  size_t source, target;
  uintptr_t source_address, target_address;
  size_t bytes;
  uint64_t token;
};

/* One software owner on one OS hart. All arrays remain live until kernel_end.
 */
struct kernel {
  struct kernel_slot *slots;
  size_t thread_count;
  const struct kernel_job *jobs;
  struct kernel_result *results;
  /* The jobs published so far, the first of them still queued. */
  size_t job_count, first, completed;
  struct kernel_transfer *transfers;
  size_t transfer_count;
  int active;
};

/* Also makes `kernel` the calling host thread's kernel_current(). Programs
 * are loaded when their threads first execute a job. */
void kernel_init(struct kernel *kernel, struct kernel_slot *slots,
                 size_t capacity);
struct kernel *kernel_current(void);
/* Dependencies name earlier jobs: every dependency index must be less than its
 * job. */
void kernel_begin(struct kernel *kernel, const struct kernel_job *jobs,
                  size_t count, struct kernel_transfer *transfers,
                  size_t transfer_count, struct kernel_result *results);
/* Publishes the jobs up to `count` (of the array kernel_begin took, which
 * published `count` of them) to a batch without transfers: the threads run
 * them while the caller records the next. */
void kernel_publish(struct kernel *kernel, size_t count);
/* Poll every running thread, acknowledge completions, then launch ready jobs,
 * each thread its jobs in order. */
size_t kernel_poll(struct kernel *kernel);
void kernel_wait(struct kernel *kernel);
void kernel_cancel(struct kernel *kernel, size_t job);
void kernel_cancel_all(struct kernel *kernel);
/* Release only after all tasks, including cancelled work, have completed. */
void kernel_end(struct kernel *kernel);
size_t kernel_run(struct kernel *kernel, const struct kernel_job *jobs,
                  size_t count, struct kernel_transfer *transfers,
                  size_t transfer_count, struct kernel_result *results);

#ifdef __cplusplus
}
#endif

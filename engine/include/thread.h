#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* OS-invisible hardware threads of the current core. */

enum thread_space { THREAD_CODE, THREAD_TLS };
enum thread_field {
  THREAD_FIELD_SIGNATURE,
  THREAD_FIELD_STATUS,
  THREAD_FIELD_CODE_BYTES,
  THREAD_FIELD_TLS_BASE,
  THREAD_FIELD_TLS_BYTES,
  THREAD_FIELD_TSS_BASE,
  THREAD_FIELD_TSS_BYTES,
  THREAD_FIELD_COUNT,
  THREAD_FIELD_COMPLETED_TASK,
  THREAD_FIELD_RETURN_VALUE
};
enum {
  THREAD_ONLINE = 1,
  THREAD_OWNED = 2,
  THREAD_DONE = 4,
  THREAD_CANCELLED = 8
};

struct thread_task {
  uint64_t id, entry, code_end, argument, stack, signature;
};

struct thread_result {
  uint64_t value;
  int cancelled;
};

uint64_t thread_query(uint32_t thread, enum thread_field field);
uint32_t thread_find(uint64_t signature, uint32_t instance);
// Data is 8-byte aligned; the last partial word is zero padded.
void thread_tss_write(uint32_t offset, const void *data, size_t bytes);
uint64_t thread_tss_read(uint32_t offset);
// Transfers require an acknowledged idle target; software owns the TSS staging
// range. Offsets and bytes are 8-byte aligned.
void thread_load(uint32_t thread, enum thread_space space, uint32_t offset,
                 uint32_t tss_offset, uint32_t bytes);
void thread_store(uint32_t thread, enum thread_space space, uint32_t offset,
                  uint32_t tss_offset, uint32_t bytes);
void thread_acquire(void);
void thread_release(void);
void thread_start(uint32_t thread, const struct thread_task *task);
void thread_cancel(uint32_t thread);
// Returns zero while running; a completed result is acknowledged before return.
int thread_poll(uint32_t thread, uint64_t task, struct thread_result *result);
uint64_t thread_wait(uint32_t thread, uint64_t task, int *cancelled);

#ifdef __cplusplus
}
#endif

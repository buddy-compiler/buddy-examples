#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ant_space { ANT_CODE, ANT_TLS, ANT_TSS };
enum ant_field {
  ANT_SIGNATURE, ANT_STATUS, ANT_CODE_BYTES, ANT_TLS_BASE, ANT_TLS_BYTES,
  ANT_TSS_BASE, ANT_TSS_BYTES, ANT_COUNT, ANT_COMPLETED_TASK, ANT_RETURN_VALUE
};
enum { ANT_ONLINE = 1, ANT_OWNED = 2, ANT_DONE = 4, ANT_CANCELLED = 8 };
struct ant_task {
  uint64_t id, entry, code_end, argument, stack, signature;
};

uint64_t ant_query(uint32_t context, enum ant_field field);
void ant_write(uint32_t context, enum ant_space space, uint32_t offset,
               const void *data, size_t bytes);
uint64_t ant_read(uint32_t context, enum ant_space space, uint32_t offset);
void ant_acquire(void);
void ant_release(void);
void ant_start(uint32_t context, const struct ant_task *task);
void ant_cancel(uint32_t context);
uint64_t ant_wait(uint32_t context, uint32_t task, int *cancelled);

#ifdef __cplusplus
}
#endif

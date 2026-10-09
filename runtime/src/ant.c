#include "ant.h"

#define COMMAND(op, key, value) ({ \
  uint64_t result; \
  __asm__ volatile(".insn r 0x2b, 7, " #op ", %0, %1, %2" \
    : "=r"(result) : "r"((uint64_t)(key)), "r"((uint64_t)(value)) : "memory"); \
  result; \
})

static uint64_t key(uint32_t context, uint32_t field) {
  return ((uint64_t)context << 32) | field;
}

uint64_t ant_query(uint32_t context, enum ant_field field) {
  return COMMAND(0, key(context, field), 0);
}

void ant_write(uint32_t context, enum ant_space space, uint32_t offset,
               const void *data, size_t bytes) {
  if (offset % 8 || bytes > UINT32_MAX - offset) __builtin_trap();
  const unsigned char *source = data;
  for (size_t i = 0; i < bytes; i += 8) {
    uint64_t word = 0;
    for (size_t byte = 0; byte < 8 && i + byte < bytes; ++byte)
      word |= (uint64_t)source[i + byte] << (8 * byte);
    uint64_t address = key(context, offset + i);
    switch (space) {
    case ANT_CODE: COMMAND(8, address, word); break;
    case ANT_TLS: COMMAND(10, address, word); break;
    case ANT_TSS: COMMAND(12, address, word); break;
    default: __builtin_trap();
    }
  }
}

uint64_t ant_read(uint32_t context, enum ant_space space, uint32_t offset) {
  uint64_t address = key(context, offset);
  switch (space) {
  case ANT_CODE: return COMMAND(7, address, 0);
  case ANT_TLS: return COMMAND(9, address, 0);
  case ANT_TSS: return COMMAND(11, address, 0);
  default: __builtin_trap();
  }
}

void ant_acquire(void) { COMMAND(5, 0, 0); }
void ant_release(void) { COMMAND(6, 0, 0); }
void ant_cancel(uint32_t context) { COMMAND(4, key(context, 0), 0); }

void ant_start(uint32_t context, const struct ant_task *task) {
  const uint64_t fields[] = {task->id, task->entry, task->code_end,
                            task->argument, task->stack, task->signature};
  for (uint32_t field = 0; field < 6; ++field)
    COMMAND(1, key(context, field), fields[field]);
  COMMAND(2, key(context, 0), 0);
}

uint64_t ant_wait(uint32_t context, uint32_t task, int *cancelled) {
  uint64_t status;
  while (!((status = ant_query(context, ANT_STATUS)) & ANT_DONE)) {}
  if (ant_query(context, ANT_COMPLETED_TASK) != task) __builtin_trap();
  uint64_t value = ant_query(context, ANT_RETURN_VALUE);
  *cancelled = !!(status & ANT_CANCELLED);
  COMMAND(3, key(context, 0), 0);
  return value;
}

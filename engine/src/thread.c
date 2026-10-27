#include "thread.h"

/* Management commands are `.insn r 0x2b, 7, <op>`: rs1 = thread << 32 | field,
 * rs2 = data. */

static uint64_t key(uint32_t thread, uint32_t field) {
  return ((uint64_t)thread << 32) | field;
}

uint64_t thread_query(uint32_t thread, enum thread_field field) {
  uint64_t value;
  __asm__ volatile(".insn r 0x2b, 7, 0, %0, %1, x0"
                   : "=r"(value)
                   : "r"(key(thread, field))
                   : "memory");
  return value;
}

uint32_t thread_find(uint64_t signature, uint32_t instance) {
  uint32_t count = thread_query(0, THREAD_FIELD_COUNT);
  for (uint32_t thread = 0; thread < count; ++thread) {
    if (thread_query(thread, THREAD_FIELD_SIGNATURE) != signature)
      continue;
    if (instance == 0)
      return thread;
    --instance;
  }
  __builtin_trap();
}

void thread_tss_write(uint32_t offset, const void *data, size_t bytes) {
  if (offset % 8 || bytes > UINT32_MAX - offset)
    __builtin_trap();
  const unsigned char *source = data;
  for (size_t i = 0; i < bytes; i += 8) {
    uint64_t word = 0;
    for (size_t byte = 0; byte < 8 && i + byte < bytes; ++byte)
      word |= (uint64_t)source[i + byte] << (8 * byte);
    __asm__ volatile(
        ".insn r 0x2b, 7, 12, x0, %0, %1" ::"r"((uint64_t)offset + i), "r"(word)
        : "memory");
  }
}

uint64_t thread_tss_read(uint32_t offset) {
  uint64_t value;
  __asm__ volatile(".insn r 0x2b, 7, 11, %0, %1, x0"
                   : "=r"(value)
                   : "r"((uint64_t)offset)
                   : "memory");
  return value;
}

void thread_load(uint32_t thread, enum thread_space space, uint32_t offset,
                 uint32_t tss_offset, uint32_t bytes) {
  if (!bytes || ((offset | tss_offset | bytes) & 7))
    __builtin_trap();
  uint64_t target = key(thread, offset);
  uint64_t transfer = ((uint64_t)bytes << 32) | tss_offset;
  if (space == THREAD_CODE)
    __asm__ volatile(".insn r 0x2b, 7, 8, x0, %0, %1" ::"r"(target),
                     "r"(transfer)
                     : "memory");
  else
    __asm__ volatile(".insn r 0x2b, 7, 10, x0, %0, %1" ::"r"(target),
                     "r"(transfer)
                     : "memory");
}

void thread_store(uint32_t thread, enum thread_space space, uint32_t offset,
                  uint32_t tss_offset, uint32_t bytes) {
  if (!bytes || ((offset | tss_offset | bytes) & 7))
    __builtin_trap();
  uint64_t target = key(thread, offset);
  uint64_t transfer = ((uint64_t)bytes << 32) | tss_offset;
  if (space == THREAD_CODE)
    __asm__ volatile(".insn r 0x2b, 7, 7, x0, %0, %1" ::"r"(target),
                     "r"(transfer)
                     : "memory");
  else
    __asm__ volatile(".insn r 0x2b, 7, 9, x0, %0, %1" ::"r"(target),
                     "r"(transfer)
                     : "memory");
}

void thread_acquire(void) {
  __asm__ volatile(".insn r 0x2b, 7, 5, x0, x0, x0" ::: "memory");
}

void thread_release(void) {
  __asm__ volatile(".insn r 0x2b, 7, 6, x0, x0, x0" ::: "memory");
}

void thread_cancel(uint32_t thread) {
  __asm__ volatile(".insn r 0x2b, 7, 4, x0, %0, x0" ::"r"(key(thread, 0))
                   : "memory");
}

void thread_start(uint32_t thread, const struct thread_task *task) {
  const uint64_t fields[] = {task->id,       task->entry, task->code_end,
                             task->argument, task->stack, task->signature};
  for (uint32_t field = 0; field < 6; ++field)
    __asm__ volatile(".insn r 0x2b, 7, 1, x0, %0, %1" ::"r"(key(thread, field)),
                     "r"(fields[field])
                     : "memory");
  __asm__ volatile(".insn r 0x2b, 7, 2, x0, %0, x0" ::"r"(key(thread, 0))
                   : "memory");
}

int thread_poll(uint32_t thread, uint64_t task, struct thread_result *result) {
  uint64_t status = thread_query(thread, THREAD_FIELD_STATUS);
  if (!(status & THREAD_DONE))
    return 0;
  if (thread_query(thread, THREAD_FIELD_COMPLETED_TASK) != task)
    __builtin_trap();
  result->value = thread_query(thread, THREAD_FIELD_RETURN_VALUE);
  result->cancelled = !!(status & THREAD_CANCELLED);
  __asm__ volatile(".insn r 0x2b, 7, 3, x0, %0, x0" ::"r"(key(thread, 0))
                   : "memory");
  return 1;
}

uint64_t thread_wait(uint32_t thread, uint64_t task, int *cancelled) {
  struct thread_result result;
  while (!thread_poll(thread, task, &result)) {
  }
  *cancelled = result.cancelled;
  return result.value;
}

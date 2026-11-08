#include "thread.h"

#include <engine_ioctl.h>

#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

/* Under Linux the OS owns the hardware threads: each function hands its
 * management instructions to the engine driver (/dev/engine), which issues
 * them in order on the CPU of the calling thread. One open file per process:
 * it owns the threads of every core the process uses. */

static int device = -1;

static void issue(struct engine_instruction *list, size_t count) {
  int file = __atomic_load_n(&device, __ATOMIC_ACQUIRE);
  if (file < 0) {
    int opened = open("/dev/engine", O_RDWR | O_CLOEXEC);
    if (opened < 0)
      abort();
    int expected = -1;
    if (__atomic_compare_exchange_n(&device, &expected, opened, 0,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
      file = opened;
    else {
      close(opened);
      file = expected;
    }
  }
  struct engine_issue request = {.count = count,
                                 .instructions = (uintptr_t)list};
  if (ioctl(file, ENGINE_ISSUE, &request))
    abort();
}

static uint64_t key(uint32_t thread, uint32_t field) {
  return ((uint64_t)thread << 32) | field;
}

static uint64_t one(uint64_t operation, uint64_t rs1, uint64_t rs2) {
  struct engine_instruction instruction = {operation, rs1, rs2, 0};
  issue(&instruction, 1);
  return instruction.rd;
}

uint64_t thread_query(uint32_t thread, enum thread_field field) {
  return one(0, key(thread, field), 0);
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
  abort();
}

typedef uint64_t __attribute__((may_alias)) tss_word;

void thread_tss_write(uint32_t offset, const void *data, size_t bytes) {
  if (offset % 8 || (uintptr_t)data % 8 || bytes > UINT32_MAX - offset)
    abort();
  const tss_word *words = data;
  size_t whole = bytes / 8, count = (bytes + 7) / 8;
  struct engine_instruction *list = malloc(count * sizeof(*list));
  if (!list)
    abort();
  for (size_t i = 0; i < whole; ++i)
    list[i] = (struct engine_instruction){12, offset + 8 * i, words[i], 0};
  if (bytes % 8) {
    const unsigned char *tail = (const unsigned char *)(words + whole);
    uint64_t word = 0;
    for (size_t byte = 0; byte < bytes % 8; ++byte)
      word |= (uint64_t)tail[byte] << (8 * byte);
    list[whole] = (struct engine_instruction){12, offset + 8 * whole, word, 0};
  }
  issue(list, count);
  free(list);
}

uint64_t thread_tss_read(uint32_t offset) { return one(11, offset, 0); }

void thread_load(uint32_t thread, enum thread_space space, uint32_t offset,
                 uint32_t tss_offset, uint32_t bytes) {
  if (!bytes || ((offset | tss_offset | bytes) & 7))
    abort();
  one(space == THREAD_CODE ? 8 : 10, key(thread, offset),
      ((uint64_t)bytes << 32) | tss_offset);
}

void thread_store(uint32_t thread, enum thread_space space, uint32_t offset,
                  uint32_t tss_offset, uint32_t bytes) {
  if (!bytes || ((offset | tss_offset | bytes) & 7))
    abort();
  one(space == THREAD_CODE ? 7 : 9, key(thread, offset),
      ((uint64_t)bytes << 32) | tss_offset);
}

void thread_acquire(void) { one(5, 0, 0); }

void thread_release(void) { one(6, 0, 0); }

void thread_cancel(uint32_t thread) { one(4, key(thread, 0), 0); }

void thread_start(uint32_t thread, const struct thread_task *task) {
  const uint64_t fields[] = {task->id,       task->entry, task->code_end,
                             task->argument, task->stack, task->signature};
  struct engine_instruction list[7];
  for (uint32_t field = 0; field < 6; ++field)
    list[field] =
        (struct engine_instruction){1, key(thread, field), fields[field], 0};
  list[6] = (struct engine_instruction){2, key(thread, 0), 0, 0};
  issue(list, 7);
}

int thread_poll(uint32_t thread, uint64_t task, struct thread_result *result) {
  uint64_t status = thread_query(thread, THREAD_FIELD_STATUS);
  if (!(status & THREAD_DONE))
    return 0;
  struct engine_instruction list[] = {
      {0, key(thread, THREAD_FIELD_COMPLETED_TASK), 0, 0},
      {0, key(thread, THREAD_FIELD_RETURN_VALUE), 0, 0},
      {3, key(thread, 0), 0, 0}};
  issue(list, 3);
  if (list[0].rd != task)
    abort();
  result->value = list[1].rd;
  result->cancelled = !!(status & THREAD_CANCELLED);
  return 1;
}

uint64_t thread_wait(uint32_t thread, uint64_t task, int *cancelled) {
  struct thread_result result;
  while (!thread_poll(thread, task, &result)) {
  }
  *cancelled = result.cancelled;
  return result.value;
}

uint64_t thread_transfer(const uint64_t descriptor[7]) {
  struct engine_instruction list[8];
  for (uint64_t field = 0; field < 7; ++field)
    list[field] = (struct engine_instruction){20, field, descriptor[field], 0};
  list[7] = (struct engine_instruction){21, 0, 0, 0};
  issue(list, 8);
  return list[7].rd;
}

void thread_transfer_ready(uint64_t token) { one(23, token, 0); }

void thread_transfer_release(uint64_t token) { one(22, token, 0); }

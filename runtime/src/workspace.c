#include "runtime.h"
#include "workspace_internal.h"
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

struct workspace;

struct block {
  size_t size;
  struct block *next, *previous;
  uint64_t available;
  struct workspace *owner;
  uint64_t reserved;
};

struct workspace {
  struct block *head;
  size_t used, peak;
  unsigned char *end;
  atomic_flag lock;
};

static struct workspace workspace = {.lock = ATOMIC_FLAG_INIT};
static _Thread_local struct workspace *current_workspace;

void *runtime_workspace_state(void) { return &workspace; }
void runtime_workspace_bind(void *space) { current_workspace = space; }

void workspace_init(void *memory, size_t bytes) {
  if ((uintptr_t)memory % 16 || bytes < sizeof(struct block))
    abort();
  // Populate user mappings before a hidden worker can allocate from this arena.
  for (size_t offset = 0; offset < bytes; offset += 4096)
    ((volatile unsigned char *)memory)[offset] = 0;
}

void workspace_begin(void *memory, size_t bytes) {
  runtime_workspace_check_idle();
  workspace.head = memory;
  *workspace.head = (struct block){.size = bytes - sizeof(struct block),
                                   .available = 1,
                                   .owner = &workspace};
  workspace.used = workspace.peak = 0;
  workspace.end = (unsigned char *)memory + bytes;
  atomic_flag_clear(&workspace.lock);
  current_workspace = &workspace;
  runtime_workspace_register(&workspace);
}

size_t workspace_peak(void) { return workspace.peak; }

void *workspace_alloc(size_t bytes) {
  struct workspace *space = current_workspace;
  if (!space || bytes > SIZE_MAX - 15)
    abort();
  bytes = (bytes + 15) & ~(size_t)15;
  while (
      atomic_flag_test_and_set_explicit(&space->lock, memory_order_acquire)) {
  }
  for (struct block *block = space->head; block; block = block->next) {
    if (!block->available || block->size < bytes)
      continue;
    if (block->size - bytes >= sizeof(struct block) + 16) {
      struct block *next =
          (struct block *)((unsigned char *)(block + 1) + bytes);
      *next = (struct block){.size = block->size - bytes - sizeof(*next),
                             .next = block->next,
                             .previous = block,
                             .available = 1,
                             .owner = space};
      if (next->next)
        next->next->previous = next;
      block->next = next;
      block->size = bytes;
    }
    block->available = 0;
    space->used += block->size + sizeof(*block);
    if (space->used > space->peak)
      space->peak = space->used;
    atomic_flag_clear_explicit(&space->lock, memory_order_release);
    return block + 1;
  }
  atomic_flag_clear_explicit(&space->lock, memory_order_release);
  fputs("task workspace exhausted\n", stderr);
  abort();
}

void workspace_free(void *memory) {
  if (!memory)
    return;
  struct workspace *space = current_workspace;
  while (
      atomic_flag_test_and_set_explicit(&space->lock, memory_order_acquire)) {
  }
  struct block *block = (struct block *)memory - 1;
  if ((uintptr_t)block < (uintptr_t)space->head ||
      (uintptr_t)memory >= (uintptr_t)space->end || block->owner != space ||
      block->available) {
    atomic_flag_clear_explicit(&space->lock, memory_order_release);
    fputs("invalid workspace free\n", stderr);
    abort();
  }
  block->available = 1;
  space->used -= block->size + sizeof(*block);
  if (block->next && block->next->available) {
    struct block *next = block->next;
    block->size += sizeof(*block) + next->size;
    block->next = next->next;
    if (block->next)
      block->next->previous = block;
  }
  if (block->previous && block->previous->available) {
    struct block *previous = block->previous;
    previous->size += sizeof(*block) + block->size;
    previous->next = block->next;
    if (previous->next)
      previous->next->previous = previous;
  }
  atomic_flag_clear_explicit(&space->lock, memory_order_release);
}

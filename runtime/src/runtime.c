#include "runtime.h"
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

#define COMMAND(op, a, b)                                                      \
  ({                                                                           \
    uintptr_t result;                                                          \
    __asm__ volatile(".insn r 0x2b, 7, " #op ", %0, %1, %2"                    \
                     : "=r"(result)                                            \
                     : "r"((uintptr_t)(a)), "r"((uintptr_t)(b))                \
                     : "memory");                                              \
    result;                                                                    \
  })

extern void *_dl_allocate_tls(void *);

struct block {
  size_t size;
  struct block *next;
  uint64_t available, reserved;
};

struct workspace {
  struct block *head;
  size_t used, peak;
  atomic_flag lock;
};

struct task {
  _Atomic uint64_t status;
  size_t core;
  task_entry entry;
  void *argument;
};

struct core {
  void *stack;
  void *tls;
  uint64_t signature;
  size_t submissions;
  task *task;
};

struct descriptor {
  uintptr_t entry, argument, stack, tls, workspace, completion;
  uint64_t signature;
};

static struct core *cores;
static size_t num_cores, stack_size;
static struct workspace workspace = {.lock = ATOMIC_FLAG_INIT};

static void execute(void *argument) {
  task *task = argument;
  task->entry(task->argument);
  __asm__ volatile(".insn r 0x7b, 3, 0, x0, x0, x0" ::: "memory");
  COMMAND(2, 0, 0);
  __builtin_unreachable();
}

void runtime_init(size_t stack_bytes) {
  num_cores = COMMAND(3, 0, 0);
  if (!num_cores || !stack_bytes || stack_bytes % 16 || cores)
    abort();
  stack_size = stack_bytes;
  cores = calloc(num_cores, sizeof(*cores));
  if (!cores)
    abort();
  for (size_t i = 0; i < num_cores; ++i) {
    cores[i].signature = COMMAND(4, i, 0);
    cores[i].stack = aligned_alloc(16, stack_size);
    cores[i].tls = _dl_allocate_tls(NULL);
    if (!cores[i].stack || !cores[i].tls)
      abort();
  }
}

size_t core_count(void) { return num_cores; }
uint64_t core_signature(size_t core) { return cores[core].signature; }

void workspace_begin(void *memory, size_t bytes) {
  for (size_t i = 0; i < num_cores; ++i)
    if (cores[i].task)
      abort();
  if ((uintptr_t)memory % 16 || bytes < sizeof(struct block))
    abort();
  workspace.head = memory;
  *workspace.head =
      (struct block){.size = bytes - sizeof(struct block), .available = 1};
  workspace.used = workspace.peak = 0;
  atomic_flag_clear(&workspace.lock);
  COMMAND(6, &workspace, 0);
}

size_t workspace_peak(void) { return workspace.peak; }

task *task_submit(uint64_t signature, task_entry entry, void *argument) {
  size_t selected = num_cores;
  int compatible = 0;
  for (;;) {
    for (size_t core = 0; core < num_cores; ++core) {
      if (cores[core].signature != signature)
        continue;
      compatible = 1;
      if (!cores[core].task || atomic_load_explicit(&cores[core].task->status,
                                                    memory_order_acquire)) {
        if (selected == num_cores || cores[core].submissions < cores[selected].submissions)
          selected = core;
      }
    }
    if (!compatible) {
      fprintf(stderr, "no compatible NPU core: required=0x%016llx; available=",
              (unsigned long long)signature);
      for (size_t core = 0; core < num_cores; ++core)
        fprintf(stderr, " %zu:0x%016llx", core, (unsigned long long)cores[core].signature);
      fputc('\n', stderr);
      abort();
    }
    if (selected != num_cores)
      break;
    sched_yield();
  }
  task *task = malloc(sizeof(*task));
  if (!task)
    abort();
  atomic_init(&task->status, 0);
  task->core = selected;
  task->entry = entry;
  task->argument = argument;
  struct core *core = &cores[selected];
  core->task = task;
  ++core->submissions;
  struct descriptor descriptor = {
      (uintptr_t)execute,
      (uintptr_t)task,
      (uintptr_t)core->stack + stack_size,
      (uintptr_t)core->tls,
      (uintptr_t)&workspace,
      (uintptr_t)&task->status,
      signature,
  };
  COMMAND(0, selected, &descriptor);
  return task;
}

int task_wait(task *task) {
  uint64_t status;
  while (!(status = atomic_load_explicit(&task->status, memory_order_acquire)))
    COMMAND(1, task->core, 0);
  if (cores[task->core].task == task)
    cores[task->core].task = NULL;
  free(task);
  return status == 1 ? 0 : -1;
}

void task_run(uint64_t signature, task_entry entry, void *argument) {
  if (task_wait(task_submit(signature, entry, argument))) {
    fputs("NPU task failed\n", stderr);
    abort();
  }
}

void *workspace_alloc(size_t bytes) {
  struct workspace *space = (struct workspace *)COMMAND(5, 0, 0);
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
                             .available = 1};
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
  struct workspace *space = (struct workspace *)COMMAND(5, 0, 0);
  while (
      atomic_flag_test_and_set_explicit(&space->lock, memory_order_acquire)) {
  }
  struct block *block = space->head;
  while (block && block + 1 != memory)
    block = block->next;
  if (!block || block->available) {
    atomic_flag_clear_explicit(&space->lock, memory_order_release);
    fputs("invalid workspace free\n", stderr);
    abort();
  }
  block->available = 1;
  space->used -= block->size + sizeof(*block);
  for (block = space->head; block && block->next;) {
    if (block->available && block->next->available) {
      block->size += sizeof(*block) + block->next->size;
      block->next = block->next->next;
    } else {
      block = block->next;
    }
  }
  atomic_flag_clear_explicit(&space->lock, memory_order_release);
}

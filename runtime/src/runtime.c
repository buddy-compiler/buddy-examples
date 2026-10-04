#include "runtime.h"
#include "workspace_internal.h"
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

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

enum { TASK_RETURN = 0xbb01 };

static struct core *cores;
static size_t num_cores, stack_size;
static void execute(void *argument) {
  runtime_workspace_bind((void *)COMMAND(5, 0, 0));
  task *task = argument;
  task->entry(task->argument);
  __asm__ volatile(".insn r 0x7b, 3, 0, x0, x0, x0" ::: "memory");
  if (COMMAND(11, 0, 0)) __builtin_trap();
  __asm__ volatile("fence rw, rw" ::: "memory");
  register uintptr_t status __asm__("a0") = 0;
  register uintptr_t operation __asm__("a7") = TASK_RETURN;
  __asm__ volatile("ecall" : "+r"(status) : "r"(operation) : "memory");
  __builtin_unreachable();
}

void runtime_init(size_t stack_bytes) {
  if (mlockall(MCL_CURRENT | MCL_FUTURE)) {
    perror("mlockall");
    abort();
  }
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
    if (cores[i].stack)
      for (size_t offset = 0; offset < stack_size; offset += 4096)
        ((volatile unsigned char *)cores[i].stack)[offset] = 0;
    if (!cores[i].stack || !cores[i].tls)
      abort();
  }
}

size_t core_count(void) { return num_cores; }
uint64_t core_signature(size_t core) { if (!core || core > num_cores) abort(); return cores[core-1].signature; }
size_t core_submissions(size_t core) { if (!core || core > num_cores) abort(); return cores[core-1].submissions; }

static task *submit(size_t required, uint64_t signature, task_entry entry, void *argument) {
  size_t selected = num_cores;
  int compatible = 0;
  for (;;) {
    for (size_t core = 0; core < num_cores; ++core) {
      if ((required != num_cores && core != required) || cores[core].signature != signature)
        continue;
      compatible = 1;
      if (cores[core].task && !atomic_load_explicit(&cores[core].task->status, memory_order_acquire)) {
        uint64_t status = COMMAND(10, core, 0);
        if (status) atomic_store_explicit(&cores[core].task->status, status, memory_order_release);
      }
      if (!cores[core].task || atomic_load_explicit(&cores[core].task->status,
                                                    memory_order_acquire)) {
        if (selected == num_cores ||
            cores[core].submissions < cores[selected].submissions)
          selected = core;
      }
    }
    if (!compatible) {
      fprintf(stderr, "no compatible NPU core: required=0x%016llx; available=",
              (unsigned long long)signature);
      for (size_t core = 0; core < num_cores; ++core)
        fprintf(stderr, " %zu:0x%016llx", core,
                (unsigned long long)cores[core].signature);
      fputc('\n', stderr);
      abort();
    }
    if (selected != num_cores)
      break;
    uint64_t status = COMMAND(1, required, signature);
    if (required == num_cores && status) abort();
    if (required != num_cores && status != 1) abort();
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
  uintptr_t gp;
  __asm__ volatile("mv %0, gp" : "=r"(gp));
  uintptr_t descriptor[] = {
      (uintptr_t)execute, (uintptr_t)task,
      (uintptr_t)core->stack + stack_size, (uintptr_t)core->tls,
      (uintptr_t)runtime_workspace_state(), gp, signature,
  };
  for (size_t field = 0; field < sizeof(descriptor) / sizeof(descriptor[0]); ++field)
    if (COMMAND(7, field, descriptor[field])) abort();
  if (COMMAND(0, selected, signature)) abort();
  return task;
}

task *task_submit(uint64_t signature, task_entry entry, void *argument) {
  return submit(num_cores, signature, entry, argument);
}
task *task_submit_on(size_t core, uint64_t signature, task_entry entry, void *argument) {
  if (!core || core > num_cores) abort();
  return submit(core - 1, signature, entry, argument);
}

int task_wait(task *task) {
  uint64_t status;
  if (!(status = atomic_load_explicit(&task->status, memory_order_acquire))) {
    status = COMMAND(1, task->core, 0);
    __asm__ volatile("fence rw, rw" ::: "memory");
    atomic_store_explicit(&task->status, status, memory_order_release);
  }
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


void task_run_on(size_t core, uint64_t signature, task_entry entry, void *argument) {
  if (task_wait(task_submit_on(core, signature, entry, argument))) abort();
}

void runtime_workspace_register(void *space) { COMMAND(6, space, 0); }
void runtime_workspace_check_idle(void) {
  for (size_t i = 0; i < num_cores; ++i)
    if (cores[i].task) abort();
}

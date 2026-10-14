#define _GNU_SOURCE
#include "runtime.h"
#include "thread_location_internal.h"
#include "workspace_internal.h"
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
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
extern void _dl_deallocate_tls(void *, bool);

struct task {
  _Atomic uint64_t status;
  size_t thread;
  task_entry entry;
  void *argument;
};

struct thread {
  void *stack;
  void *tls;
  uint64_t signature;
  size_t submissions;
  task *task;
};

enum { TASK_RETURN = 0xbb01 };

static struct thread *threads;
static size_t num_threads, stack_size, outstanding;
static uint32_t controller;
static _Thread_local size_t current_thread;
static _Thread_local int controller_thread;
static void execute(void *argument) {
  runtime_workspace_bind((void *)COMMAND(5, 0, 0));
  task *task = argument;
  current_thread = task->thread + 1;
  task->entry(task->argument);
  if (COMMAND(11, 0, 0))
    __builtin_trap();
  __asm__ volatile("fence rw, rw" ::: "memory");
  register uintptr_t status __asm__("a0") = 0;
  register uintptr_t operation __asm__("a7") = TASK_RETURN;
  __asm__ volatile("ecall" : "+r"(status) : "r"(operation) : "memory");
  __builtin_unreachable();
}

void runtime_init(size_t stack_bytes) {
  controller = runtime_cpu_id();
  controller_thread = 1;
  if (mlockall(MCL_CURRENT | MCL_FUTURE)) {
    perror("mlockall");
    abort();
  }
  num_threads = COMMAND(3, 0, 0);
  if (!num_threads || !stack_bytes || stack_bytes % 16 || threads)
    abort();
  stack_size = stack_bytes;
  threads = calloc(num_threads, sizeof(*threads));
  if (!threads)
    abort();
  for (size_t i = 0; i < num_threads; ++i) {
    threads[i].signature = COMMAND(4, i, 0);
    threads[i].stack = aligned_alloc(16, stack_size);
    threads[i].tls = _dl_allocate_tls(NULL);
    if (threads[i].stack)
      for (size_t offset = 0; offset < stack_size; offset += 4096)
        ((volatile unsigned char *)threads[i].stack)[offset] = 0;
    if (!threads[i].stack || !threads[i].tls)
      abort();
  }
}

size_t thread_count(void) { return num_threads; }
thread_location_t thread_location(void) {
  if (threads && (controller_thread || current_thread))
    return (thread_location_t){controller, current_thread};
  return (thread_location_t){runtime_cpu_id(), 0};
}
void runtime_shutdown(void) {
  if (!threads || outstanding)
    abort();
  for (size_t i = 0; i < num_threads; ++i) {
    if (threads[i].task)
      abort();
    _dl_deallocate_tls(threads[i].tls, true);
    free(threads[i].stack);
  }
  free(threads);
  threads = NULL;
  num_threads = stack_size = 0;
  controller_thread = 0;
  runtime_workspace_bind(NULL);
}
uint64_t thread_signature(size_t thread) {
  if (!thread || thread > num_threads)
    abort();
  return threads[thread - 1].signature;
}
size_t thread_submissions(size_t thread) {
  if (!thread || thread > num_threads)
    abort();
  return threads[thread - 1].submissions;
}

static task *submit(size_t required, uint64_t signature, task_entry entry,
                    void *argument) {
  size_t selected = num_threads;
  int compatible = 0;
  for (;;) {
    for (size_t thread = 0; thread < num_threads; ++thread) {
      if ((required != num_threads && thread != required) ||
          threads[thread].signature != signature)
        continue;
      compatible = 1;
      if (threads[thread].task &&
          !atomic_load_explicit(&threads[thread].task->status,
                                memory_order_acquire)) {
        uint64_t status = COMMAND(10, thread, 0);
        if (status)
          atomic_store_explicit(&threads[thread].task->status, status,
                                memory_order_release);
      }
      if (!threads[thread].task ||
          atomic_load_explicit(&threads[thread].task->status,
                               memory_order_acquire)) {
        if (selected == num_threads ||
            threads[thread].submissions < threads[selected].submissions)
          selected = thread;
      }
    }
    if (!compatible) {
      fprintf(stderr,
              "no compatible NPU thread: required=0x%016llx; available=",
              (unsigned long long)signature);
      for (size_t thread = 0; thread < num_threads; ++thread)
        fprintf(stderr, " %zu:0x%016llx", thread,
                (unsigned long long)threads[thread].signature);
      fputc('\n', stderr);
      abort();
    }
    if (selected != num_threads)
      break;
    if (sched_yield())
      abort();
  }
  task *task = malloc(sizeof(*task));
  if (!task)
    abort();
  atomic_init(&task->status, 0);
  task->thread = selected;
  task->entry = entry;
  task->argument = argument;
  struct thread *thread = &threads[selected];
  thread->task = task;
  ++outstanding;
  ++thread->submissions;
  uintptr_t gp;
  __asm__ volatile("mv %0, gp" : "=r"(gp));
  uintptr_t descriptor[] = {
      (uintptr_t)execute,
      (uintptr_t)task,
      (uintptr_t)thread->stack + stack_size,
      (uintptr_t)thread->tls,
      (uintptr_t)runtime_workspace_state(),
      gp,
      signature,
  };
  for (size_t field = 0; field < sizeof(descriptor) / sizeof(descriptor[0]);
       ++field)
    if (COMMAND(7, field, descriptor[field]))
      abort();
  if (COMMAND(0, selected, signature))
    abort();
  return task;
}

task *task_submit(uint64_t signature, task_entry entry, void *argument) {
  return submit(num_threads, signature, entry, argument);
}
task *task_submit_on(size_t thread, uint64_t signature, task_entry entry,
                     void *argument) {
  if (!thread || thread > num_threads)
    abort();
  return submit(thread - 1, signature, entry, argument);
}

int task_wait(task *task) {
  uint64_t status;
  if (!(status = atomic_load_explicit(&task->status, memory_order_acquire))) {
    while (!(status = COMMAND(10, task->thread, 0)))
      if (sched_yield())
        abort();
    __asm__ volatile("fence rw, rw" ::: "memory");
    atomic_store_explicit(&task->status, status, memory_order_release);
  }
  if (threads[task->thread].task == task)
    threads[task->thread].task = NULL;
  --outstanding;
  free(task);
  return status == 1 ? 0 : -1;
}

void task_run(uint64_t signature, task_entry entry, void *argument) {
  if (task_wait(task_submit(signature, entry, argument))) {
    fputs("NPU task failed\n", stderr);
    abort();
  }
}

void task_run_on(size_t thread, uint64_t signature, task_entry entry,
                 void *argument) {
  if (task_wait(task_submit_on(thread, signature, entry, argument)))
    abort();
}

void runtime_workspace_register(void *space) { COMMAND(6, space, 0); }
void runtime_workspace_check_idle(void) {
  for (size_t i = 0; i < num_threads; ++i)
    if (threads[i].task)
      abort();
}

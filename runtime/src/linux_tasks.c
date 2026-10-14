#define _GNU_SOURCE
#include "params.h"
#include "runtime.h"
#include "thread_location_internal.h"
#include "workspace_internal.h"
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <topology.h>

struct task {
  task_entry entry;
  void *argument;
  int done;
};
struct thread {
  unsigned cpu;
  pthread_t thread;
  task *job;
  size_t submissions;
};
static struct thread *threads;
static size_t num_threads;
static size_t outstanding;
static int stopping;
static void *workspace;
static pthread_t controller;
static uint32_t controller_cpu;
static _Thread_local size_t current_thread;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t changed = PTHREAD_COND_INITIALIZER;

static void check(int result) {
  if (result) {
    fprintf(stderr, "Linux task runtime error: %d\n", result);
    abort();
  }
}

static void *execute(void *argument) {
  struct thread *thread = argument;
  current_thread = (size_t)(thread - threads) + 1;
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  CPU_SET(thread->cpu, &affinity);
  check(pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity));
  check(pthread_mutex_lock(&lock));
  for (;;) {
    while (!thread->job && !stopping)
      check(pthread_cond_wait(&changed, &lock));
    if (stopping)
      break;
    task *job = thread->job;
    check(pthread_mutex_unlock(&lock));
    runtime_workspace_bind(workspace);
    job->entry(job->argument);
#ifdef __riscv
    __asm__ volatile(".insn r 0x7b, 3, 0, x0, x0, x0" ::: "memory");
#endif
    check(pthread_mutex_lock(&lock));
    job->done = 1;
    thread->job = NULL;
    check(pthread_cond_broadcast(&changed));
  }
  check(pthread_mutex_unlock(&lock));
  return NULL;
}

void runtime_init(size_t stack_bytes) {
  if (threads)
    abort();
  controller = pthread_self();
  controller_cpu = runtime_cpu_id();
  workspace = runtime_workspace_state();
  cpu_set_t affinity;
  if (sched_getaffinity(0, sizeof(affinity), &affinity))
    abort();
  const uint32_t profile =
      bb_topology_thread_profile(bb_topology_thread_id(BB_TEST_HART));
  for (unsigned cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    if (cpu >= BB_MAIN_THREADS ||
        bb_topology_thread_profile((thread_id_t){0, cpu}) != profile)
      CPU_CLR(cpu, &affinity);
  num_threads = CPU_COUNT(&affinity);
  if (!num_threads)
    abort();
  threads = calloc(num_threads, sizeof(*threads));
  if (!threads)
    abort();
  pthread_attr_t attributes;
  check(pthread_attr_init(&attributes));
  check(pthread_attr_setstacksize(&attributes, stack_bytes));
  size_t index = 0;
  for (unsigned cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &affinity))
      continue;
    threads[index].cpu = cpu;
    check(pthread_create(&threads[index].thread, &attributes, execute,
                         &threads[index]));
    ++index;
  }
  check(pthread_attr_destroy(&attributes));
}

size_t thread_count(void) { return num_threads; }
thread_location_t thread_location(void) {
  if (threads && (current_thread || pthread_equal(controller, pthread_self())))
    return (thread_location_t){controller_cpu, current_thread};
  return (thread_location_t){runtime_cpu_id(), 0};
}
void runtime_shutdown(void) {
  if (!threads || !pthread_equal(controller, pthread_self()))
    abort();
  check(pthread_mutex_lock(&lock));
  if (outstanding)
    abort();
  for (size_t i = 0; i < num_threads; ++i)
    if (threads[i].job)
      abort();
  stopping = 1;
  check(pthread_cond_broadcast(&changed));
  check(pthread_mutex_unlock(&lock));
  for (size_t i = 0; i < num_threads; ++i)
    check(pthread_join(threads[i].thread, NULL));
  free(threads);
  threads = NULL;
  num_threads = 0;
  stopping = 0;
  workspace = NULL;
  runtime_workspace_bind(NULL);
}
uint64_t thread_signature(size_t thread) {
  if (!thread || thread > num_threads)
    abort();
  return THREAD_SIGNATURE;
}
size_t thread_submissions(size_t thread) {
  if (!thread || thread > num_threads)
    abort();
  return threads[thread - 1].submissions;
}

static task *submit(size_t required, uint64_t signature, task_entry entry,
                    void *argument) {
  if (signature != THREAD_SIGNATURE)
    abort();
  task *job = malloc(sizeof(*job));
  if (!job)
    abort();
  *job = (task){entry, argument, 0};
  check(pthread_mutex_lock(&lock));
  for (;;) {
    size_t selected = num_threads;
    for (size_t i = 0; i < num_threads; ++i)
      if ((required == num_threads || i == required) && !threads[i].job &&
          (selected == num_threads ||
           threads[i].submissions < threads[selected].submissions))
        selected = i;
    if (selected != num_threads) {
      threads[selected].job = job;
      ++threads[selected].submissions;
      ++outstanding;
      check(pthread_cond_broadcast(&changed));
      check(pthread_mutex_unlock(&lock));
      return job;
    }
    check(pthread_cond_wait(&changed, &lock));
  }
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
int task_wait(task *job) {
  check(pthread_mutex_lock(&lock));
  while (!job->done)
    check(pthread_cond_wait(&changed, &lock));
  --outstanding;
  check(pthread_mutex_unlock(&lock));
  free(job);
  return 0;
}
void task_run(uint64_t signature, task_entry entry, void *argument) {
  task_wait(task_submit(signature, entry, argument));
}
void task_run_on(size_t thread, uint64_t signature, task_entry entry,
                 void *argument) {
  task_wait(task_submit_on(thread, signature, entry, argument));
}
void runtime_workspace_register(void *space) {
  check(pthread_mutex_lock(&lock));
  workspace = space;
  check(pthread_mutex_unlock(&lock));
}
void runtime_workspace_check_idle(void) {
  check(pthread_mutex_lock(&lock));
  for (size_t i = 0; i < num_threads; ++i)
    if (threads[i].job)
      abort();
  check(pthread_mutex_unlock(&lock));
}

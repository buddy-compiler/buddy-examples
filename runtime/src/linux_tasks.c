#define _GNU_SOURCE
#include "runtime.h"
#include "workspace_internal.h"
#include "params.h"
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <stdio.h>

struct task { task_entry entry; void *argument; int done; };
struct core { unsigned cpu; pthread_t thread; task *job; size_t submissions; };
static struct core *cores;
static size_t num_cores;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t changed = PTHREAD_COND_INITIALIZER;

static void check(int result) {
  if (result) { fprintf(stderr, "Linux task runtime error: %d\n", result); abort(); }
}

static void *execute(void *argument) {
  struct core *core = argument;
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  CPU_SET(core->cpu, &affinity);
  check(pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity));
  check(pthread_mutex_lock(&lock));
  for (;;) {
    while (!core->job) check(pthread_cond_wait(&changed, &lock));
    task *job = core->job;
    check(pthread_mutex_unlock(&lock));
    runtime_workspace_bind(runtime_workspace_state());
    job->entry(job->argument);
#ifdef __riscv
    __asm__ volatile(".insn r 0x7b, 3, 0, x0, x0, x0" ::: "memory");
#endif
    check(pthread_mutex_lock(&lock));
    job->done = 1;
    core->job = NULL;
    check(pthread_cond_broadcast(&changed));
  }
}

void runtime_init(size_t stack_bytes) {
  if (cores) abort();
  cpu_set_t affinity;
  if (sched_getaffinity(0, sizeof(affinity), &affinity)) abort();
  num_cores = CPU_COUNT(&affinity);
  if (!num_cores) abort();
  cores = calloc(num_cores, sizeof(*cores));
  if (!cores) abort();
  pthread_attr_t attributes;
  check(pthread_attr_init(&attributes));
  check(pthread_attr_setstacksize(&attributes, stack_bytes));
  size_t index = 0;
  for (unsigned cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &affinity)) continue;
    cores[index].cpu = cpu;
    check(pthread_create(&cores[index].thread, &attributes, execute, &cores[index]));
    ++index;
  }
  check(pthread_attr_destroy(&attributes));
}

size_t core_count(void) { return num_cores; }
uint64_t core_signature(size_t core) { if (!core || core > num_cores) abort(); return CORE_SIGNATURE; }
size_t core_submissions(size_t core) { if (!core || core > num_cores) abort(); return cores[core-1].submissions; }

static task *submit(size_t required, uint64_t signature, task_entry entry, void *argument) {
  if (signature != CORE_SIGNATURE) abort();
  task *job = malloc(sizeof(*job));
  if (!job) abort();
  *job = (task){entry, argument, 0};
  check(pthread_mutex_lock(&lock));
  for (;;) {
    size_t selected = num_cores;
    for (size_t i = 0; i < num_cores; ++i)
      if ((required == num_cores || i == required) && !cores[i].job && (selected == num_cores || cores[i].submissions < cores[selected].submissions)) selected = i;
    if (selected != num_cores) {
      cores[selected].job = job;
      ++cores[selected].submissions;
      check(pthread_cond_broadcast(&changed));
      check(pthread_mutex_unlock(&lock));
      return job;
    }
    check(pthread_cond_wait(&changed, &lock));
  }
}

task *task_submit(uint64_t signature, task_entry entry, void *argument) {
  return submit(num_cores, signature, entry, argument);
}
task *task_submit_on(size_t core, uint64_t signature, task_entry entry, void *argument) {
  if (!core || core > num_cores) abort();
  return submit(core - 1, signature, entry, argument);
}
int task_wait(task *job) {
  check(pthread_mutex_lock(&lock));
  while (!job->done) check(pthread_cond_wait(&changed, &lock));
  check(pthread_mutex_unlock(&lock));
  free(job);
  return 0;
}
void task_run(uint64_t signature, task_entry entry, void *argument) {
  task_wait(task_submit(signature, entry, argument));
}
void task_run_on(size_t core, uint64_t signature, task_entry entry, void *argument) {
  task_wait(task_submit_on(core, signature, entry, argument));
}
void runtime_workspace_register(void *space) { (void)space; }
void runtime_workspace_check_idle(void) {
  check(pthread_mutex_lock(&lock));
  for (size_t i = 0; i < num_cores; ++i) if (cores[i].job) abort();
  check(pthread_mutex_unlock(&lock));
}

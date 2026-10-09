#define _GNU_SOURCE
#include "runtime.h"
#include "workspace_internal.h"
#include "ant.h"
#include "ant_stream.h"
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/mman.h>

struct instruction { uint64_t operation, rs1, rs2; };
struct batch { uint64_t count; struct instruction instructions[]; };
struct task { task_entry entry; void *argument; int done; };
struct core {
  pthread_t thread;
  task *job;
  struct batch *batch;
  size_t capacity, submissions;
  uint64_t signature, base, stack;
  uint32_t context, sequence;
};
static struct core *cores;
static size_t num_cores;
static unsigned controller;
static int stopping;
static _Thread_local struct core *current;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t changed = PTHREAD_COND_INITIALIZER;
extern const unsigned char ant_stream_image[], ant_stream_image_end[];
static void check(int status) { if (status) { fprintf(stderr, "Ant runtime error: %d\n", status); abort(); } }

void ant_flush(void) {
  if (!current) abort();
  struct core *core = current;
  if (!core->batch->count) return;
  if (++core->sequence == 0) abort();
  ant_write(core->context, ANT_TLS, 0, core->batch,
            sizeof(uint64_t) + core->batch->count * sizeof(struct instruction));
  struct ant_task job = {core->sequence, 0, ant_stream_image_end - ant_stream_image,
                        core->base, core->stack, core->signature};
  ant_start(core->context, &job);
  int cancelled;
  if (ant_wait(core->context, core->sequence, &cancelled) || cancelled) abort();
  core->batch->count = 0;
}

void ant_emit(uint32_t funct7, uint64_t rs1, uint64_t rs2) {
  if (!current || funct7 > 127) abort();
  if (current->batch->count == current->capacity) ant_flush();
  current->batch->instructions[current->batch->count++] = (struct instruction){funct7, rs1, rs2};
  // Returning from a DMA call guarantees that CPU reuse of its DDR range is safe.
  // Bank-only operations remain batched; no additional fence instruction is inserted.
  if (funct7 == 0 || funct7 == 12 || funct7 == 16 || (funct7 >= 33 && funct7 <= 35)) ant_flush();
}

static void *execute(void *argument) {
  current = argument;
  cpu_set_t affinity;
  CPU_ZERO(&affinity); CPU_SET(controller, &affinity);
  check(pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity));
  check(pthread_mutex_lock(&lock));
  for (;;) {
    while (!current->job && !stopping) check(pthread_cond_wait(&changed, &lock));
    if (stopping) break;
    task *job = current->job;
    check(pthread_mutex_unlock(&lock));
    runtime_workspace_bind(runtime_workspace_state());
    job->entry(job->argument);
    ant_flush();
    check(pthread_mutex_lock(&lock));
    job->done = 1;
    current->job = NULL;
    check(pthread_cond_broadcast(&changed));
  }
  check(pthread_mutex_unlock(&lock));
  return NULL;
}

static void shutdown_runtime(void) {
  check(pthread_mutex_lock(&lock));
  for (size_t i = 0; i < num_cores; ++i) if (cores[i].job) abort();
  stopping = 1;
  check(pthread_cond_broadcast(&changed));
  check(pthread_mutex_unlock(&lock));
  for (size_t i = 0; i < num_cores; ++i) {
    check(pthread_join(cores[i].thread, NULL));
    free(cores[i].batch);
  }
  ant_release();
  free(cores);
}

void runtime_init(size_t stack_bytes) {
  if (cores || mlockall(MCL_CURRENT | MCL_FUTURE)) abort();
  cpu_set_t affinity;
  if (sched_getaffinity(0, sizeof(affinity), &affinity) || CPU_COUNT(&affinity) != 1) abort();
  controller = sched_getcpu();
  num_cores = ant_query(0, ANT_COUNT);
  if (!num_cores) abort();
  cores = calloc(num_cores, sizeof(*cores));
  if (!cores) abort();
  size_t image_bytes = ant_stream_image_end - ant_stream_image;
  for (size_t i = 0; i < num_cores; ++i) {
    struct core *core = &cores[i];
    uint64_t bytes = ant_query(i, ANT_TLS_BYTES);
    if (ant_query(i, ANT_STATUS) != ANT_ONLINE || bytes <= 512 + sizeof(uint64_t) ||
        image_bytes > ant_query(i, ANT_CODE_BYTES)) abort();
    core->context = i;
    core->signature = ant_query(i, ANT_SIGNATURE);
    core->base = ant_query(i, ANT_TLS_BASE);
    core->stack = core->base + bytes;
    core->capacity = (bytes - 512 - sizeof(uint64_t)) / sizeof(struct instruction);
    core->batch = malloc(sizeof(uint64_t) + core->capacity * sizeof(struct instruction));
    if (!core->batch) abort();
    core->batch->count = 0;
    ant_write(i, ANT_CODE, 0, ant_stream_image, image_bytes);
  }
  ant_acquire();
  pthread_attr_t attributes;
  check(pthread_attr_init(&attributes));
  check(pthread_attr_setstacksize(&attributes, stack_bytes));
  for (size_t i = 0; i < num_cores; ++i) check(pthread_create(&cores[i].thread, &attributes, execute, &cores[i]));
  check(pthread_attr_destroy(&attributes));
  if (atexit(shutdown_runtime)) abort();
}

size_t core_count(void) { return num_cores; }
uint64_t core_signature(size_t core) { if (!core || core > num_cores) abort(); return cores[core-1].signature; }
size_t core_submissions(size_t core) { if (!core || core > num_cores) abort(); return cores[core-1].submissions; }
static task *submit(size_t required, uint64_t signature, task_entry entry, void *argument) {
  if (current) abort();
  task *job = malloc(sizeof(*job));
  if (!job) abort();
  *job = (task){entry, argument, 0};
  check(pthread_mutex_lock(&lock));
  for (;;) {
    size_t selected = num_cores;
    int compatible = 0;
    for (size_t i = 0; i < num_cores; ++i) {
      if ((required != num_cores && i != required) || cores[i].signature != signature) continue;
      compatible = 1;
      if (!cores[i].job && (selected == num_cores || cores[i].submissions < cores[selected].submissions)) selected = i;
    }
    if (!compatible) abort();
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
task *task_submit(uint64_t signature, task_entry entry, void *argument) { return submit(num_cores, signature, entry, argument); }
task *task_submit_on(size_t core, uint64_t signature, task_entry entry, void *argument) {
  if (!core || core > num_cores) abort();
  return submit(core-1, signature, entry, argument);
}
int task_wait(task *job) {
  check(pthread_mutex_lock(&lock));
  while (!job->done) check(pthread_cond_wait(&changed, &lock));
  check(pthread_mutex_unlock(&lock));
  free(job);
  return 0;
}
void task_run(uint64_t signature, task_entry entry, void *argument) { task_wait(task_submit(signature, entry, argument)); }
void task_run_on(size_t core, uint64_t signature, task_entry entry, void *argument) { task_wait(task_submit_on(core, signature, entry, argument)); }
void runtime_workspace_register(void *space) { (void)space; }
void runtime_workspace_check_idle(void) {
  check(pthread_mutex_lock(&lock));
  for (size_t i = 0; i < num_cores; ++i) if (cores[i].job) abort();
  check(pthread_mutex_unlock(&lock));
}

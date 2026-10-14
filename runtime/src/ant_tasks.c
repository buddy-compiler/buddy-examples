#define _GNU_SOURCE
#include "ant.h"
#include "ant_stream.h"
#include "runtime.h"
#include "thread_location_internal.h"
#include "workspace_internal.h"
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

struct instruction {
  uint64_t operation, rs1, rs2;
};
struct batch {
  uint64_t count;
  struct instruction instructions[];
};
struct runtime;
struct task {
  task_entry entry;
  void *argument;
  int done;
  struct runtime *owner;
};
struct thread {
  struct runtime *owner;
  pthread_t thread;
  task *job;
  struct batch *batch;
  size_t capacity, submissions;
  uint64_t signature, base, stack;
  uint32_t context, sequence;
};
struct runtime {
  struct thread *threads;
  size_t num_threads, outstanding;
  unsigned controller;
  int stopping;
  void *workspace;
  pthread_mutex_t lock;
  pthread_cond_t changed;
};
static _Thread_local struct runtime owned = {
    .lock = PTHREAD_MUTEX_INITIALIZER, .changed = PTHREAD_COND_INITIALIZER};
static _Thread_local struct runtime *owner;
static _Thread_local struct thread *current;
static pthread_once_t process_init = PTHREAD_ONCE_INIT;
static _Atomic size_t active_instances;
uint32_t thread_batches(size_t thread) {
  return owner->threads[thread - 1].sequence;
}
extern const unsigned char ant_stream_image[], ant_stream_image_end[];
static void check(int status) {
  if (status) {
    fprintf(stderr, "Ant runtime error: %d\n", status);
    abort();
  }
}

void ant_flush(void) {
  if (!current)
    abort();
  struct thread *thread = current;
  if (!thread->batch->count)
    return;
  if (++thread->sequence == 0)
    abort();
  ant_write(thread->context, ANT_TLS, 0, thread->batch,
            sizeof(uint64_t) +
                thread->batch->count * sizeof(struct instruction));
  struct ant_task job = {thread->sequence,
                         0,
                         ant_stream_image_end - ant_stream_image,
                         thread->base,
                         thread->stack,
                         thread->signature};
  ant_start(thread->context, &job);
  int cancelled;
  if (ant_wait(thread->context, thread->sequence, &cancelled) || cancelled)
    abort();
  thread->batch->count = 0;
}

void ant_emit(uint32_t funct7, uint64_t rs1, uint64_t rs2) {
  if (!current || funct7 == 0 || funct7 > 127)
    abort();
  if (current->batch->count == current->capacity)
    ant_flush();
  current->batch->instructions[current->batch->count++] =
      (struct instruction){funct7, rs1, rs2};
  // Returning from a DMA call guarantees that CPU reuse of its DDR range is
  // safe. Bank-only operations remain batched.
  if (funct7 == 44 || funct7 == 16 || (funct7 >= 33 && funct7 <= 35))
    ant_flush();
}

static void *execute(void *argument) {
  current = argument;
  owner = current->owner;
  struct runtime *runtime = owner;
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  CPU_SET(runtime->controller, &affinity);
  check(pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity));
  check(pthread_mutex_lock(&runtime->lock));
  for (;;) {
    while (!current->job && !runtime->stopping)
      check(pthread_cond_wait(&runtime->changed, &runtime->lock));
    if (runtime->stopping)
      break;
    task *job = current->job;
    check(pthread_mutex_unlock(&runtime->lock));
    runtime_workspace_bind(runtime->workspace);
    job->entry(job->argument);
    ant_flush();
    check(pthread_mutex_lock(&runtime->lock));
    job->done = 1;
    current->job = NULL;
    check(pthread_cond_broadcast(&runtime->changed));
  }
  check(pthread_mutex_unlock(&runtime->lock));
  return NULL;
}

void runtime_shutdown(void) {
  struct runtime *runtime = owner;
  if (!runtime || runtime != &owned || current || !runtime->threads)
    abort();
  cpu_set_t affinity;
  if (sched_getaffinity(0, sizeof(affinity), &affinity) ||
      CPU_COUNT(&affinity) != 1 || !CPU_ISSET(runtime->controller, &affinity))
    abort();
  check(pthread_mutex_lock(&runtime->lock));
  if (runtime->outstanding)
    abort();
  for (size_t i = 0; i < runtime->num_threads; ++i)
    if (runtime->threads[i].job)
      abort();
  runtime->stopping = 1;
  check(pthread_cond_broadcast(&runtime->changed));
  check(pthread_mutex_unlock(&runtime->lock));
  for (size_t i = 0; i < runtime->num_threads; ++i) {
    check(pthread_join(runtime->threads[i].thread, NULL));
    free(runtime->threads[i].batch);
  }
  ant_release();
  free(runtime->threads);
  runtime->threads = NULL;
  runtime->num_threads = 0;
  runtime->workspace = NULL;
  runtime->stopping = 0;
  owner = NULL;
  runtime_workspace_bind(NULL);
  atomic_fetch_sub(&active_instances, 1);
}

static void shutdown_at_exit(void) {
  if (owner == &owned && owned.threads && !current)
    runtime_shutdown();
  if (atomic_load(&active_instances))
    abort();
}

static void initialize_process(void) {
  if (mlockall(MCL_CURRENT | MCL_FUTURE) || atexit(shutdown_at_exit))
    abort();
}

void runtime_init(size_t stack_bytes) {
  if (owner || current)
    abort();
  check(pthread_once(&process_init, initialize_process));
  owner = &owned;
  struct runtime *runtime = owner;
  runtime->workspace = runtime_workspace_state();
  cpu_set_t affinity;
  if (sched_getaffinity(0, sizeof(affinity), &affinity) ||
      CPU_COUNT(&affinity) != 1)
    abort();
  runtime->controller = sched_getcpu();
  runtime->num_threads = ant_query(0, ANT_COUNT);
  if (!runtime->num_threads)
    abort();
  runtime->threads = calloc(runtime->num_threads, sizeof(*runtime->threads));
  if (!runtime->threads)
    abort();
  size_t image_bytes = ant_stream_image_end - ant_stream_image;
  for (size_t i = 0; i < runtime->num_threads; ++i) {
    struct thread *thread = &runtime->threads[i];
    thread->owner = runtime;
    uint64_t bytes = ant_query(i, ANT_TLS_BYTES);
    if (ant_query(i, ANT_STATUS) != ANT_ONLINE ||
        bytes <= 512 + sizeof(uint64_t) ||
        image_bytes > ant_query(i, ANT_CODE_BYTES))
      abort();
    thread->context = i;
    thread->signature = ant_query(i, ANT_SIGNATURE);
    thread->base = ant_query(i, ANT_TLS_BASE);
    thread->stack = thread->base + bytes;
    thread->capacity =
        (bytes - 512 - sizeof(uint64_t)) / sizeof(struct instruction);
    thread->batch = malloc(sizeof(uint64_t) +
                           thread->capacity * sizeof(struct instruction));
    if (!thread->batch)
      abort();
    thread->batch->count = 0;
    ant_write(i, ANT_CODE, 0, ant_stream_image, image_bytes);
  }
  ant_acquire();
  pthread_attr_t attributes;
  check(pthread_attr_init(&attributes));
  check(pthread_attr_setstacksize(&attributes, stack_bytes));
  for (size_t i = 0; i < runtime->num_threads; ++i)
    check(pthread_create(&runtime->threads[i].thread, &attributes, execute,
                         &runtime->threads[i]));
  check(pthread_attr_destroy(&attributes));
  atomic_fetch_add(&active_instances, 1);
}

size_t thread_count(void) {
  if (!owner)
    abort();
  return owner->num_threads;
}
thread_location_t thread_location(void) {
  if (owner)
    return (thread_location_t){owner->controller,
                               current ? current->context + 1 : 0};
  return (thread_location_t){runtime_cpu_id(), 0};
}
uint64_t thread_signature(size_t thread) {
  if (!owner || !thread || thread > owner->num_threads)
    abort();
  return owner->threads[thread - 1].signature;
}
size_t thread_submissions(size_t thread) {
  if (!owner || !thread || thread > owner->num_threads)
    abort();
  return owner->threads[thread - 1].submissions;
}
static task *submit(size_t required, uint64_t signature, task_entry entry,
                    void *argument) {
  if (current || !owner || owner != &owned)
    abort();
  struct runtime *runtime = owner;
  task *job = malloc(sizeof(*job));
  if (!job)
    abort();
  *job = (task){entry, argument, 0, runtime};
  check(pthread_mutex_lock(&runtime->lock));
  for (;;) {
    size_t selected = runtime->num_threads;
    int compatible = 0;
    for (size_t i = 0; i < runtime->num_threads; ++i) {
      if ((required != runtime->num_threads && i != required) ||
          runtime->threads[i].signature != signature)
        continue;
      compatible = 1;
      if (!runtime->threads[i].job &&
          (selected == runtime->num_threads ||
           runtime->threads[i].submissions <
               runtime->threads[selected].submissions))
        selected = i;
    }
    if (!compatible)
      abort();
    if (selected != runtime->num_threads) {
      runtime->threads[selected].job = job;
      ++runtime->threads[selected].submissions;
      ++runtime->outstanding;
      check(pthread_cond_broadcast(&runtime->changed));
      check(pthread_mutex_unlock(&runtime->lock));
      return job;
    }
    check(pthread_cond_wait(&runtime->changed, &runtime->lock));
  }
}
task *task_submit(uint64_t signature, task_entry entry, void *argument) {
  return submit(thread_count(), signature, entry, argument);
}
task *task_submit_on(size_t thread, uint64_t signature, task_entry entry,
                     void *argument) {
  if (!thread || thread > thread_count())
    abort();
  return submit(thread - 1, signature, entry, argument);
}
int task_wait(task *job) {
  struct runtime *runtime = job->owner;
  check(pthread_mutex_lock(&runtime->lock));
  while (!job->done)
    check(pthread_cond_wait(&runtime->changed, &runtime->lock));
  --runtime->outstanding;
  check(pthread_mutex_unlock(&runtime->lock));
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
  if (!owner || owner != &owned || current)
    abort();
  check(pthread_mutex_lock(&owner->lock));
  owner->workspace = space;
  check(pthread_mutex_unlock(&owner->lock));
}
void runtime_workspace_check_idle(void) {
  if (!owner || owner != &owned || current)
    abort();
  check(pthread_mutex_lock(&owner->lock));
  for (size_t i = 0; i < owner->num_threads; ++i)
    if (owner->threads[i].job)
      abort();
  check(pthread_mutex_unlock(&owner->lock));
}

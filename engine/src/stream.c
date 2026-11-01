#include "stream.h"

#include <stdlib.h>
#include <string.h>

static _Thread_local struct engine *engine;
static _Thread_local struct stream *buffer;
extern _Thread_local uint32_t stream_target;
static _Thread_local size_t capacity;
static _Thread_local uint64_t sequence;
// Inside stream_job: the instructions are recorded, not run.
static _Thread_local int recording;
static _Thread_local size_t loop_depth, loop_start[STREAM_LOOP_DEPTH];
struct stream_allocation {
  void *address;
  struct stream_allocation *next;
};
static _Thread_local struct stream_allocation *allocations;
struct stream_copy_request {
  unsigned char *destination;
  const unsigned char *source;
  uint64_t rows, bytes, destination_stride, source_stride;
  struct stream_copy_request *next;
};
static _Thread_local struct stream_copy_request *copies;

// The open batch of this core; its arrays and streams are freed by
// stream_wait.
static _Thread_local struct {
  struct engine_job *jobs;
  struct engine_result *results;
  struct engine_transfer *transfers;
  size_t *dependencies;
  size_t job_count, dependency_count, transfer_count;
  size_t jobs_end, dependencies_end, transfers_end;
  int submitted;
  // The core's mcycle when it submitted the batch.
  uint64_t submit;
} batch;

#ifndef ENGINE_LINUX
static uint64_t mcycle(void) {
  uint64_t value;
  __asm__ volatile("csrr %0, mcycle" : "=r"(value));
  return value;
}
#endif

void stream_flush(void) {
  if (!engine || recording)
    __builtin_trap();
  if (!buffer->count)
    return;
  const struct engine_job job = {
      .thread = stream_target,
      .task = {.id = ++sequence,
               .signature = engine->slots[stream_target].signature},
      .opaque = buffer,
  };
  // The thread returns its cycles in the batch.
  struct engine_result result;
  if (engine_run(engine, &job, 1, 0, 0, &result))
    __builtin_trap();
  buffer->count = 0;
}

void stream_emit(uint32_t funct7, uint64_t rs1, uint64_t rs2) {
  if (!engine || funct7 == 0 || funct7 > 127)
    __builtin_trap();
  if (buffer->count == capacity) {
    // A recorded job is one stream: it must fit the thread.
    if (recording)
      __builtin_trap();
    stream_flush();
  }
  buffer->instructions[buffer->count++] =
      (struct stream_instruction){funct7, rs1, rs2};
  // Returning from a DMA instruction guarantees that host reuse of its DDR
  // range is safe; bank-only instructions stay batched. A recorded job runs
  // later as a whole: the host leaves its buffers alone until stream_wait.
  if (!recording &&
      (funct7 == 44 || funct7 == 16 || (funct7 >= 33 && funct7 <= 35)))
    stream_flush();
}

void stream_loop(uint64_t count) {
  if (!recording || loop_depth == STREAM_LOOP_DEPTH ||
      buffer->count == capacity)
    __builtin_trap();
  loop_start[loop_depth++] = buffer->count;
  buffer->instructions[buffer->count++] =
      (struct stream_instruction){STREAM_LOOP, count, 0};
}

void stream_end(void) {
  if (!recording || !loop_depth || buffer->count == capacity)
    __builtin_trap();
  size_t start = loop_start[--loop_depth];
  buffer->instructions[buffer->count++] =
      (struct stream_instruction){STREAM_END, start, 0};
  buffer->instructions[start].rs2 = buffer->count;
}

void stream_offset(uint64_t depth, uint64_t bytes) {
  if (!recording || depth >= loop_depth || buffer->count == capacity)
    __builtin_trap();
  buffer->instructions[buffer->count++] =
      (struct stream_instruction){STREAM_OFFSET, depth, bytes};
}

void stream_copy(void *destination, const void *source, uint64_t rows,
                 uint64_t bytes, uint64_t destination_stride,
                 uint64_t source_stride) {
  if (!recording)
    __builtin_trap();
  struct stream_copy_request *copy = malloc(sizeof(*copy));
  if (!copy)
    __builtin_trap();
  *copy = (struct stream_copy_request){
      destination,        source,        rows,  bytes,
      destination_stride, source_stride, copies};
  copies = copy;
}

void stream_release(void *allocation) {
  if (!recording)
    __builtin_trap();
  struct stream_allocation *node = malloc(sizeof(*node));
  if (!node)
    __builtin_trap();
  *node = (struct stream_allocation){allocation, allocations};
  allocations = node;
}

// The instructions of one batch for `thread` of `owner`: its TLS (below its
// stack), staged through TSS.
static size_t stream_bytes(const struct engine_slot *slot) {
  return slot->tls_bytes - STREAM_STACK_BYTES < slot->tss_bytes
             ? slot->tls_bytes - STREAM_STACK_BYTES
             : slot->tss_bytes;
}

void stream_run(uint64_t signature, void (*entry)(void *), void *argument) {
  if (engine)
    __builtin_trap();
  struct engine *owner = engine_current();
  uint32_t thread = 0;
  while (thread < owner->thread_count &&
         !(owner->slots[thread].online &&
           owner->slots[thread].signature == signature))
    ++thread;
  if (thread == owner->thread_count)
    __builtin_trap();
  size_t bytes = stream_bytes(&owner->slots[thread]);
  uint64_t storage[bytes / sizeof(uint64_t)];
  engine = owner;
  stream_target = thread;
  buffer = (struct stream *)storage;
  capacity = (bytes - sizeof(uint64_t)) / sizeof(struct stream_instruction);
  buffer->count = 0;
  entry(argument);
  stream_flush();
  engine = 0;
  stream_target = UINT32_MAX;
}

void stream_begin(size_t jobs, size_t dependencies, size_t transfers) {
  if (engine || batch.jobs || !jobs)
    __builtin_trap();
  batch.jobs = malloc(jobs * sizeof(struct engine_job));
  batch.results = malloc(jobs * sizeof(struct engine_result));
  batch.transfers = malloc(transfers * sizeof(struct engine_transfer));
  batch.dependencies = malloc(dependencies * sizeof(size_t));
  if (!batch.jobs || !batch.results || (transfers && !batch.transfers) ||
      (dependencies && !batch.dependencies))
    __builtin_trap();
  batch.job_count = batch.dependency_count = batch.transfer_count = 0;
  batch.jobs_end = jobs;
  batch.dependencies_end = dependencies;
  batch.transfers_end = transfers;
  batch.submitted = 0;
}

void stream_job(uint64_t signature, uint32_t thread, void (*entry)(void *),
                void *argument) {
  if (engine || !batch.jobs || batch.submitted ||
      batch.job_count == batch.jobs_end)
    __builtin_trap();
  struct engine *owner = engine_current();
  if (thread >= owner->thread_count || !owner->slots[thread].online ||
      owner->slots[thread].signature != signature)
    __builtin_trap();
  size_t bytes = stream_bytes(&owner->slots[thread]);
  uint64_t storage[bytes / sizeof(uint64_t)];
  engine = owner;
  stream_target = thread;
  buffer = (struct stream *)storage;
  capacity = (bytes - sizeof(uint64_t)) / sizeof(struct stream_instruction);
  buffer->count = 0;
  recording = 1;
  entry(argument);
  if (loop_depth)
    __builtin_trap();
  recording = 0;
  engine = 0;
  stream_target = UINT32_MAX;
  size_t recorded =
      sizeof(uint64_t) + buffer->count * sizeof(struct stream_instruction);
  struct stream *copy = malloc(recorded);
  if (!copy)
    __builtin_trap();
  memcpy(copy, buffer, recorded);
  batch.jobs[batch.job_count++] = (struct engine_job){
      .thread = thread,
      .task = {.id = ++sequence, .signature = signature},
      .dependencies = batch.dependencies + batch.dependency_count,
      .opaque = copy};
}

void stream_after(size_t dependency) {
  if (!batch.job_count || batch.submitted ||
      dependency + 1 >= batch.job_count ||
      batch.dependency_count == batch.dependencies_end)
    __builtin_trap();
  batch.dependencies[batch.dependency_count++] = dependency;
  ++batch.jobs[batch.job_count - 1].dependency_count;
}

void stream_transfer(size_t source, size_t target_job, void *source_address,
                     void *target_address, size_t bytes) {
  if (!batch.jobs || batch.submitted ||
      batch.transfer_count == batch.transfers_end)
    __builtin_trap();
  batch.transfers[batch.transfer_count++] =
      (struct engine_transfer){.source = source,
                               .target = target_job,
                               .source_address = (uintptr_t)source_address,
                               .target_address = (uintptr_t)target_address,
                               .bytes = bytes};
}

void stream_submit(void) {
  if (!batch.jobs || batch.submitted || batch.job_count != batch.jobs_end ||
      batch.dependency_count != batch.dependencies_end ||
      batch.transfer_count != batch.transfers_end)
    __builtin_trap();
#ifndef ENGINE_LINUX
  batch.submit = mcycle();
#endif
  engine_begin(engine_current(), batch.jobs, batch.job_count, batch.transfers,
               batch.transfer_count, batch.results);
  batch.submitted = 1;
}

/* The core's clock (mcycle) and its threads' are virtual, so that the same
 * program gives the same latency however the simulator interleaves the core
 * and its threads: the core does not count its spinning here, and moves its
 * clock to the end of the last job. Each thread runs the jobs given to it in
 * the order of the batch; a job starts once the batch is submitted, its
 * thread is free and the jobs it depends on or receives from have ended, and
 * lasts the thread's own cycles in it, which the stream interpreter measures
 * with mcycle and returns as the job's value. Under Linux (ENGINE_LINUX) the
 * clocks are the OS's: machine mode alone writes mcycle and mcountinhibit,
 * so stream_wait only waits. */
void stream_wait(void) {
  if (!batch.submitted)
    __builtin_trap();
  struct engine *owner = engine_current();
#ifdef ENGINE_LINUX
  engine_wait(owner);
  for (size_t job = 0; job < batch.job_count; ++job)
    if (batch.results[job].state != ENGINE_SUCCEEDED)
      __builtin_trap();
#else
  uint64_t now = mcycle();
  __asm__ volatile("csrsi 0x320, 1" ::: "memory");
  engine_wait(owner);
  uint64_t ends[batch.job_count];
  uint64_t idle[owner->thread_count];
  for (size_t thread = 0; thread < owner->thread_count; ++thread)
    idle[thread] = batch.submit;
  for (size_t job = 0; job < batch.job_count; ++job) {
    const struct engine_job *request = &batch.jobs[job];
    const struct engine_result *result = &batch.results[job];
    if (result->state != ENGINE_SUCCEEDED)
      __builtin_trap();
    uint64_t start = idle[request->thread];
    for (size_t i = 0; i < request->dependency_count; ++i)
      if (ends[request->dependencies[i]] > start)
        start = ends[request->dependencies[i]];
    for (size_t i = 0; i < batch.transfer_count; ++i)
      if (batch.transfers[i].target == job &&
          ends[batch.transfers[i].source] > start)
        start = ends[batch.transfers[i].source];
    ends[job] = start + result->value;
    idle[request->thread] = ends[job];
    if (ends[job] > now)
      now = ends[job];
  }
  __asm__ volatile("csrw mcycle, %0\n"
                   "csrci 0x320, 1" ::"r"(now)
                   : "memory");
#endif
  engine_end(owner);
  while (copies) {
    struct stream_copy_request *copy = copies;
    copies = copy->next;
    for (uint64_t row = 0; row < copy->rows; ++row)
      memcpy(copy->destination + row * copy->destination_stride,
             copy->source + row * copy->source_stride, copy->bytes);
    free(copy);
  }
  while (allocations) {
    struct stream_allocation *node = allocations;
    allocations = node->next;
    free(node->address);
    free(node);
  }
  for (size_t job = 0; job < batch.job_count; ++job)
    free(batch.jobs[job].opaque);
  free(batch.jobs);
  free(batch.results);
  free(batch.transfers);
  free(batch.dependencies);
  batch.jobs = 0;
  batch.submitted = 0;
}

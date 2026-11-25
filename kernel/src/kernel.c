#include "kernel.h"

extern const struct kernel_binding __start_kernel_bindings[],
    __stop_kernel_bindings[];

static _Thread_local struct kernel *current;

struct kernel *kernel_current(void) {
  if (!current)
    __builtin_trap();
  return current;
}

void kernel_init(struct kernel *kernel, struct kernel_slot *slots,
                 size_t capacity) {
  uint64_t count = thread_query(0, THREAD_FIELD_COUNT);
  if (count > capacity || count > UINT32_MAX)
    __builtin_trap();
  *kernel = (struct kernel){.slots = slots, .thread_count = count};
  current = kernel;
  for (uint32_t thread = 0; thread < count; ++thread) {
    struct kernel_slot *slot = &slots[thread];
    uint64_t status = thread_query(thread, THREAD_FIELD_STATUS);
    if (status & (THREAD_OWNED | THREAD_DONE))
      __builtin_trap();
    *slot = (struct kernel_slot){
        .signature = thread_query(thread, THREAD_FIELD_SIGNATURE),
        .code_bytes = thread_query(thread, THREAD_FIELD_CODE_BYTES),
        .tls_base = thread_query(thread, THREAD_FIELD_TLS_BASE),
        .tls_bytes = thread_query(thread, THREAD_FIELD_TLS_BYTES),
        .tss_base = thread_query(thread, THREAD_FIELD_TSS_BASE),
        .tss_bytes = thread_query(thread, THREAD_FIELD_TSS_BYTES),
        .online = !!(status & THREAD_ONLINE),
        .job = SIZE_MAX};
    for (const struct kernel_binding *binding = __start_kernel_bindings;
         binding < __stop_kernel_bindings; ++binding)
      if (binding->signature == slot->signature)
        slot->binding = binding;
    if (!slot->binding)
      __builtin_trap();
  }
}

void kernel_begin(struct kernel *kernel, const struct kernel_job *jobs,
                  size_t count, struct kernel_transfer *transfers,
                  size_t transfer_count, struct kernel_result *results) {
  if (kernel->active || kernel->thread_count > 64)
    __builtin_trap();
  // A transfer names its jobs by task id: the ids increase along the batch.
  if (transfer_count)
    for (size_t job = 1; job < count; ++job)
      if (jobs[job].task.id <= jobs[job - 1].task.id)
        __builtin_trap();
  kernel->jobs = jobs;
  kernel->results = results;
  kernel->job_count = 0;
  kernel->first = 0;
  kernel->completed = 0;
  kernel->transfers = transfers;
  kernel->transfer_count = transfer_count;
  thread_acquire();
  for (size_t i = 0; i < transfer_count; ++i) {
    struct kernel_transfer *edge = &transfers[i];
    if (edge->source >= edge->target || edge->target >= count ||
        jobs[edge->source].thread == KERNEL_ANY_THREAD ||
        jobs[edge->target].thread == KERNEL_ANY_THREAD)
      __builtin_trap();
    edge->token = UINT64_MAX;
    if (!edge->bytes || jobs[edge->source].empty || jobs[edge->target].empty)
      continue;
    const uint64_t descriptor[] = {jobs[edge->source].thread,
                                   jobs[edge->source].task.id,
                                   edge->source_address,
                                   jobs[edge->target].thread,
                                   jobs[edge->target].task.id,
                                   edge->target_address,
                                   edge->bytes};
    edge->token = thread_transfer(descriptor);
  }
  kernel->active = 1;
  kernel_publish(kernel, count);
}

void kernel_publish(struct kernel *kernel, size_t count) {
  if (!kernel->active || count < kernel->job_count ||
      (kernel->transfer_count && kernel->job_count))
    __builtin_trap();
  for (size_t job = kernel->job_count; job < count; ++job) {
    const struct kernel_job *request = &kernel->jobs[job];
    for (size_t i = 0; i < request->dependency_count; ++i)
      if (request->dependencies[i] >= job)
        __builtin_trap();
    int supported = 0;
    for (size_t thread = 0; thread < kernel->thread_count; ++thread) {
      const struct kernel_slot *slot = &kernel->slots[thread];
      if (slot->online && slot->signature == request->task.signature &&
          (request->thread == KERNEL_ANY_THREAD || request->thread == thread))
        supported = 1;
    }
    if (!supported)
      __builtin_trap();
    kernel->results[job] = (struct kernel_result){.state = KERNEL_QUEUED,
                                                  .thread = KERNEL_ANY_THREAD};
  }
  kernel->job_count = count;
  kernel_poll(kernel);
}

size_t kernel_poll(struct kernel *kernel) {
  if (!kernel->active)
    __builtin_trap();
  for (uint32_t thread = 0; thread < kernel->thread_count; ++thread) {
    struct kernel_slot *slot = &kernel->slots[thread];
    if (slot->job == SIZE_MAX)
      continue;
    struct thread_result completion;
    if (!thread_poll(thread, kernel->jobs[slot->job].task.id, &completion))
      continue;
    struct kernel_result *result = &kernel->results[slot->job];
    result->state = completion.cancelled ? KERNEL_CANCELLED : KERNEL_SUCCEEDED;
    result->value = completion.value;
    if (!completion.cancelled)
      for (size_t i = 0; i < kernel->transfer_count; ++i)
        if (kernel->transfers[i].token != UINT64_MAX &&
            kernel->transfers[i].source == slot->job)
          thread_transfer_ready(kernel->transfers[i].token);
    slot->job = SIZE_MAX;
    ++kernel->completed;
  }
  // Each thread runs its jobs in their order: a thread's first queued job
  // holds the ones after it, and the scan stops once every thread is held.
  while (kernel->first < kernel->job_count &&
         kernel->results[kernel->first].state != KERNEL_QUEUED)
    ++kernel->first;
  uint64_t all = kernel->thread_count == 64
                     ? UINT64_MAX
                     : (UINT64_C(1) << kernel->thread_count) - 1;
  uint64_t held = 0;
  for (uint32_t thread = 0; thread < kernel->thread_count; ++thread)
    if (kernel->slots[thread].job != SIZE_MAX || !kernel->slots[thread].online)
      held |= UINT64_C(1) << thread;
  for (size_t job = kernel->first; job < kernel->job_count && held != all;
       ++job) {
    struct kernel_result *result = &kernel->results[job];
    if (result->state != KERNEL_QUEUED)
      continue;
    const struct kernel_job *request = &kernel->jobs[job];
    if (request->thread != KERNEL_ANY_THREAD) {
      uint64_t mine = UINT64_C(1) << request->thread;
      if (held & mine)
        continue;
      held |= mine;
    }
    int ready = 1, cancelled = 0;
    for (size_t i = 0; i < request->dependency_count; ++i) {
      enum kernel_state state = kernel->results[request->dependencies[i]].state;
      ready &= state == KERNEL_SUCCEEDED;
      cancelled |= state == KERNEL_CANCELLED;
    }
    for (size_t i = 0; i < kernel->transfer_count; ++i) {
      if (kernel->transfers[i].target != job)
        continue;
      enum kernel_state state =
          kernel->results[kernel->transfers[i].source].state;
      ready &= state == KERNEL_SUCCEEDED;
      cancelled |= state == KERNEL_CANCELLED;
    }
    if (cancelled) {
      result->state = KERNEL_CANCELLED;
      ++kernel->completed;
      continue;
    }
    if (!ready)
      continue;
    if (request->empty) {
      result->state = KERNEL_SUCCEEDED;
      result->value = 0;
      ++kernel->completed;
      continue;
    }
    for (uint32_t thread = 0; thread < kernel->thread_count; ++thread) {
      struct kernel_slot *slot = &kernel->slots[thread];
      if (!slot->online || slot->job != SIZE_MAX ||
          slot->signature != request->task.signature ||
          (request->thread != KERNEL_ANY_THREAD && request->thread != thread))
        continue;
      struct thread_task task = request->task;
      if (!slot->loaded) {
        slot->binding->load(thread);
        slot->loaded = 1;
      }
      slot->binding->launch(thread, request->opaque, &task);
      slot->job = job;
      result->thread = thread;
      result->state = KERNEL_RUNNING;
      break;
    }
  }
  return kernel->job_count - kernel->completed;
}

void kernel_wait(struct kernel *kernel) {
  while (kernel_poll(kernel)) {
    if (thread_query(0, THREAD_FIELD_WAIT_COMPLETION) != THREAD_WAIT_DONE)
      __builtin_trap();
  }
}

void kernel_cancel(struct kernel *kernel, size_t job) {
  if (!kernel->active || job >= kernel->job_count)
    __builtin_trap();
  struct kernel_result *result = &kernel->results[job];
  if (result->state == KERNEL_QUEUED) {
    result->state = KERNEL_CANCELLED;
    ++kernel->completed;
  } else if (result->state == KERNEL_RUNNING) {
    thread_cancel(result->thread);
    result->state = KERNEL_CANCELLING;
  }
}

void kernel_cancel_all(struct kernel *kernel) {
  if (!kernel->active)
    __builtin_trap();
  for (size_t job = 0; job < kernel->job_count; ++job)
    kernel_cancel(kernel, job);
}

void kernel_end(struct kernel *kernel) {
  if (!kernel->active || kernel->completed != kernel->job_count)
    __builtin_trap();
  for (size_t i = 0; i < kernel->transfer_count; ++i)
    if (kernel->transfers[i].token != UINT64_MAX)
      thread_transfer_release(kernel->transfers[i].token);
  thread_release();
  kernel->active = 0;
}

size_t kernel_run(struct kernel *kernel, const struct kernel_job *jobs,
                  size_t count, struct kernel_transfer *transfers,
                  size_t transfer_count, struct kernel_result *results) {
  kernel_begin(kernel, jobs, count, transfers, transfer_count, results);
  kernel_wait(kernel);
  size_t cancelled = 0;
  for (size_t job = 0; job < count; ++job)
    cancelled += results[job].state == KERNEL_CANCELLED;
  kernel_end(kernel);
  return cancelled;
}

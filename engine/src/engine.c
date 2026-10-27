#include "engine.h"

extern const struct engine_binding __start_engine_bindings[],
    __stop_engine_bindings[];

static _Thread_local struct engine *current;

struct engine *engine_current(void) {
  if (!current)
    __builtin_trap();
  return current;
}

void engine_init(struct engine *engine, struct engine_slot *slots,
                 size_t capacity) {
  uint64_t count = thread_query(0, THREAD_FIELD_COUNT);
  if (count > capacity || count > UINT32_MAX)
    __builtin_trap();
  *engine = (struct engine){.slots = slots, .thread_count = count};
  current = engine;
  for (uint32_t thread = 0; thread < count; ++thread) {
    struct engine_slot *slot = &slots[thread];
    uint64_t status = thread_query(thread, THREAD_FIELD_STATUS);
    if (status & (THREAD_OWNED | THREAD_DONE))
      __builtin_trap();
    *slot = (struct engine_slot){
        .signature = thread_query(thread, THREAD_FIELD_SIGNATURE),
        .code_bytes = thread_query(thread, THREAD_FIELD_CODE_BYTES),
        .tls_base = thread_query(thread, THREAD_FIELD_TLS_BASE),
        .tls_bytes = thread_query(thread, THREAD_FIELD_TLS_BYTES),
        .tss_base = thread_query(thread, THREAD_FIELD_TSS_BASE),
        .tss_bytes = thread_query(thread, THREAD_FIELD_TSS_BYTES),
        .online = !!(status & THREAD_ONLINE),
        .job = SIZE_MAX};
    for (const struct engine_binding *binding = __start_engine_bindings;
         binding < __stop_engine_bindings; ++binding)
      if (binding->signature == slot->signature)
        slot->binding = binding;
    if (!slot->binding)
      __builtin_trap();
  }
}

void engine_begin(struct engine *engine, const struct engine_job *jobs,
                  size_t count, struct engine_result *results) {
  if (engine->active)
    __builtin_trap();
  for (size_t job = 0; job < count; ++job) {
    const struct engine_job *request = &jobs[job];
    for (size_t i = 0; i < request->dependency_count; ++i)
      if (request->dependencies[i] >= job)
        __builtin_trap();
    int supported = 0;
    for (size_t thread = 0; thread < engine->thread_count; ++thread) {
      const struct engine_slot *slot = &engine->slots[thread];
      if (slot->online && slot->signature == request->task.signature &&
          (request->thread == ENGINE_ANY_THREAD || request->thread == thread))
        supported = 1;
    }
    if (!supported)
      __builtin_trap();
  }
  engine->jobs = jobs;
  engine->results = results;
  engine->job_count = count;
  engine->completed = 0;
  for (size_t job = 0; job < count; ++job)
    results[job] = (struct engine_result){.state = ENGINE_QUEUED,
                                          .thread = ENGINE_ANY_THREAD};
  thread_acquire();
  engine->active = 1;
  engine_poll(engine);
}

size_t engine_poll(struct engine *engine) {
  if (!engine->active)
    __builtin_trap();
  for (uint32_t thread = 0; thread < engine->thread_count; ++thread) {
    struct engine_slot *slot = &engine->slots[thread];
    if (slot->job == SIZE_MAX)
      continue;
    struct thread_result completion;
    if (!thread_poll(thread, engine->jobs[slot->job].task.id, &completion))
      continue;
    struct engine_result *result = &engine->results[slot->job];
    result->state = completion.cancelled ? ENGINE_CANCELLED : ENGINE_SUCCEEDED;
    result->value = completion.value;
    slot->job = SIZE_MAX;
    ++engine->completed;
  }
  for (size_t job = 0; job < engine->job_count; ++job) {
    struct engine_result *result = &engine->results[job];
    if (result->state != ENGINE_QUEUED)
      continue;
    const struct engine_job *request = &engine->jobs[job];
    int ready = 1, cancelled = 0;
    for (size_t i = 0; i < request->dependency_count; ++i) {
      enum engine_state state = engine->results[request->dependencies[i]].state;
      ready &= state == ENGINE_SUCCEEDED;
      cancelled |= state == ENGINE_CANCELLED;
    }
    if (cancelled) {
      result->state = ENGINE_CANCELLED;
      ++engine->completed;
      continue;
    }
    if (!ready)
      continue;
    for (uint32_t thread = 0; thread < engine->thread_count; ++thread) {
      struct engine_slot *slot = &engine->slots[thread];
      if (!slot->online || slot->job != SIZE_MAX ||
          slot->signature != request->task.signature ||
          (request->thread != ENGINE_ANY_THREAD && request->thread != thread))
        continue;
      struct thread_task task = request->task;
      slot->binding->prepare(thread, request->opaque, &task);
      thread_start(thread, &task);
      slot->job = job;
      result->thread = thread;
      result->state = ENGINE_RUNNING;
      break;
    }
  }
  return engine->job_count - engine->completed;
}

void engine_wait(struct engine *engine) {
  while (engine_poll(engine)) {
  }
}

void engine_cancel(struct engine *engine, size_t job) {
  if (!engine->active || job >= engine->job_count)
    __builtin_trap();
  struct engine_result *result = &engine->results[job];
  if (result->state == ENGINE_QUEUED) {
    result->state = ENGINE_CANCELLED;
    ++engine->completed;
  } else if (result->state == ENGINE_RUNNING) {
    thread_cancel(result->thread);
    result->state = ENGINE_CANCELLING;
  }
}

void engine_cancel_all(struct engine *engine) {
  if (!engine->active)
    __builtin_trap();
  for (size_t job = 0; job < engine->job_count; ++job)
    engine_cancel(engine, job);
}

void engine_end(struct engine *engine) {
  if (!engine->active || engine->completed != engine->job_count)
    __builtin_trap();
  thread_release();
  engine->active = 0;
}

size_t engine_run(struct engine *engine, const struct engine_job *jobs,
                  size_t count, struct engine_result *results) {
  engine_begin(engine, jobs, count, results);
  engine_wait(engine);
  size_t cancelled = 0;
  for (size_t job = 0; job < count; ++job)
    cancelled += results[job].state == ENGINE_CANCELLED;
  engine_end(engine);
  return cancelled;
}

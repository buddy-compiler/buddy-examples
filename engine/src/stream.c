#include "stream.h"

static _Thread_local struct engine *engine;
static _Thread_local struct stream *buffer;
static _Thread_local uint32_t target;
static _Thread_local size_t capacity;
static _Thread_local uint64_t sequence;

void stream_flush(void) {
  if (!engine)
    __builtin_trap();
  if (!buffer->count)
    return;
  const struct engine_job job = {
      .thread = target,
      .task = {.id = ++sequence, .signature = engine->slots[target].signature},
      .opaque = buffer,
  };
  struct engine_result result;
  if (engine_run(engine, &job, 1, &result) || result.value)
    __builtin_trap();
  buffer->count = 0;
}

void stream_emit(uint32_t funct7, uint64_t rs1, uint64_t rs2) {
  if (!engine || funct7 == 0 || funct7 > 127)
    __builtin_trap();
  if (buffer->count == capacity)
    stream_flush();
  buffer->instructions[buffer->count++] =
      (struct stream_instruction){funct7, rs1, rs2};
  // Returning from a DMA instruction guarantees that host reuse of its DDR
  // range is safe; bank-only instructions stay batched.
  if (funct7 == 44 || funct7 == 16 || (funct7 >= 33 && funct7 <= 35))
    stream_flush();
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
  // A batch lives in the thread's TLS (below its stack) and is staged through
  // TSS.
  const struct engine_slot *slot = &owner->slots[thread];
  size_t bytes = slot->tls_bytes - 256 < slot->tss_bytes ? slot->tls_bytes - 256
                                                         : slot->tss_bytes;
  uint64_t storage[bytes / sizeof(uint64_t)];
  engine = owner;
  target = thread;
  buffer = (struct stream *)storage;
  capacity = (bytes - sizeof(uint64_t)) / sizeof(struct stream_instruction);
  buffer->count = 0;
  entry(argument);
  stream_flush();
  engine = 0;
}

uint32_t stream_thread(void) { return engine ? target + 1 : 0; }

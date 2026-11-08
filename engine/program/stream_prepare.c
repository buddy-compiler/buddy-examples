#include "stream.h"

static const unsigned char image[] __attribute__((aligned(8))) = {
#embed "program.bin"
};

/* Loads the stream interpreter into a thread, once. */
static void load(uint32_t thread) {
  size_t code_bytes = sizeof(image);
  if (code_bytes > thread_query(thread, THREAD_FIELD_CODE_BYTES) ||
      code_bytes > thread_query(thread, THREAD_FIELD_TSS_BYTES))
    __builtin_trap();
  thread_tss_write(0, image, code_bytes);
  thread_load(thread, THREAD_CODE, 0, 0, (code_bytes + 7) & ~(size_t)7);
}

/* Hands a thread one instruction batch. */
static void prepare(uint32_t thread, void *opaque, struct thread_task *task) {
  size_t data_bytes =
      sizeof(uint64_t) + ((const struct stream *)opaque)->count *
                             sizeof(struct stream_instruction);
  uint64_t tls_base = thread_query(thread, THREAD_FIELD_TLS_BASE);
  uint64_t tls_bytes = thread_query(thread, THREAD_FIELD_TLS_BYTES);
  if (((data_bytes + 7) & ~(size_t)7) + STREAM_STACK_BYTES > tls_bytes ||
      data_bytes > thread_query(thread, THREAD_FIELD_TSS_BYTES))
    __builtin_trap();
  thread_tss_write(0, opaque, data_bytes);
  thread_load(thread, THREAD_TLS, 0, 0, (data_bytes + 7) & ~(size_t)7);
  task->entry = 0;
  task->code_end = sizeof(image);
  task->argument = tls_base;
  task->stack = tls_base + tls_bytes;
}

static const struct engine_binding binding
    __attribute__((section("engine_bindings"), used)) = {THREAD_SIGNATURE, load,
                                                         prepare};

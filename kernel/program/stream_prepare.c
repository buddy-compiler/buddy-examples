#include "stream.h"

static const unsigned char image[] __attribute__((aligned(8))) = {
#embed "program.bin"
};

/* Loads the stream interpreter into a thread, once. */
static void load(uint32_t thread) {
  const struct kernel_slot *slot = &kernel_current()->slots[thread];
  size_t code_bytes = sizeof(image);
  if (code_bytes > slot->code_bytes || code_bytes > slot->tss_bytes)
    __builtin_trap();
  thread_tss_write(0, image, code_bytes);
  thread_load(thread, THREAD_CODE, 0, 0, (code_bytes + 7) & ~(size_t)7);
}

/* Hands a thread one instruction batch. */
static void launch(uint32_t thread, void *opaque, struct thread_task *task) {
  const struct kernel_slot *slot = &kernel_current()->slots[thread];
  size_t data_bytes =
      sizeof(uint64_t) + ((const struct stream *)opaque)->count *
                             sizeof(struct stream_instruction);
  if (((data_bytes + 7) & ~(size_t)7) + STREAM_STACK_BYTES > slot->tls_bytes ||
      data_bytes > slot->tss_bytes)
    __builtin_trap();
  task->entry = 0;
  task->code_end = sizeof(image);
  task->argument = slot->tls_base;
  task->stack = slot->tls_base + slot->tls_bytes;
  thread_start_data(thread, task, opaque, data_bytes);
}

static const struct kernel_binding binding
    __attribute__((section("kernel_bindings"), used)) = {THREAD_SIGNATURE, load,
                                                         launch};

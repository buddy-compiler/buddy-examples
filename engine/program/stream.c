#include "stream.h"

static uint64_t mcycle(void) {
  uint64_t value;
  __asm__ volatile("csrr %0, mcycle" : "=r"(value));
  return value;
}

/* Executes one batch of Ball instructions and returns the thread's own
 * cycles in it (mcycle), which the core reads as the job's duration. */
uint64_t entry(const struct stream *batch) {
  uint64_t start = mcycle();
  uint64_t iteration[STREAM_LOOP_DEPTH], depth = 0, offset = 0;
  for (uint64_t i = 0; i < batch->count; ++i) {
    const struct stream_instruction *encoded = &batch->instructions[i];
    if (encoded->operation == STREAM_LOOP) {
      if (!encoded->rs1) {
        i = encoded->rs2 - 1;
        continue;
      }
      if (depth == STREAM_LOOP_DEPTH)
        __builtin_trap();
      iteration[depth++] = 0;
      continue;
    }
    if (encoded->operation == STREAM_END) {
      if (!depth)
        __builtin_trap();
      if (++iteration[depth - 1] < batch->instructions[encoded->rs1].rs1)
        i = encoded->rs1;
      else
        --depth;
      continue;
    }
    if (encoded->operation == STREAM_OFFSET) {
      if (encoded->rs1 >= depth)
        __builtin_trap();
      offset += iteration[encoded->rs1] * encoded->rs2;
      continue;
    }
    struct stream_instruction instruction = *encoded;
    instruction.rs2 += offset;
    offset = 0;
    const struct stream_instruction *op = &instruction;
    switch (op->operation) {
#include "instructions.inc"
    default:
      __builtin_trap();
    }
  }
  // The Balls finish the stream's work after its last instruction issues:
  // the fence waits for them, so the duration is the whole job.
  __asm__ volatile("fence" ::: "memory");
  return mcycle() - start;
}

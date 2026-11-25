#include "stream.h"

static uint64_t mcycle(void) {
  uint64_t value;
  __asm__ volatile("csrr %0, mcycle" : "=r"(value));
  return value;
}

/* Issues Ball instruction funct7 with rs1 and rs2 (start.S). */
void issue(uint64_t funct7, uint64_t rs1, uint64_t rs2);

/* Executes one batch of Ball instructions and returns the thread's own
 * cycles in it (mcycle), which the hart reads as the job's duration. Ball
 * instructions take the shortest path: they are most of the stream. */
uint64_t entry(const struct stream *batch) {
  uint64_t start = mcycle();
  uint64_t iteration[STREAM_LOOP_DEPTH], depth = 0, offset1 = 0, offset2 = 0;
  const struct stream_instruction *first = batch->instructions;
  const struct stream_instruction *end = first + batch->count;
  for (const struct stream_instruction *encoded = first; encoded < end;
       ++encoded) {
    uint64_t operation = encoded->operation;
    if (operation < STREAM_LOOP) {
      issue(operation, encoded->rs1 + offset1, encoded->rs2 + offset2);
      offset1 = offset2 = 0;
      continue;
    }
    if (operation == STREAM_LOOP) {
      if (!encoded->rs1) {
        encoded = first + encoded->rs2 - 1;
        continue;
      }
      if (depth == STREAM_LOOP_DEPTH)
        __builtin_trap();
      iteration[depth++] = 0;
      continue;
    }
    if (operation == STREAM_END) {
      if (!depth)
        __builtin_trap();
      if (++iteration[depth - 1] < first[encoded->rs1].rs1)
        encoded = first + encoded->rs1;
      else
        --depth;
      continue;
    }
    if (encoded->rs1 >= depth)
      __builtin_trap();
    if (operation == STREAM_OFFSET_1)
      offset1 += iteration[encoded->rs1] * encoded->rs2;
    else if (operation == STREAM_OFFSET_2)
      offset2 += iteration[encoded->rs1] * encoded->rs2;
    else
      __builtin_trap();
  }
  // The Balls finish the stream's work after its last instruction issues:
  // the fence waits for them, so the duration is the whole job.
  __asm__ volatile("fence" ::: "memory");
  return mcycle() - start;
}

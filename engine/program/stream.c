#include "stream.h"

uint64_t entry(const struct stream *batch) {
  for (uint64_t i = 0; i < batch->count; ++i) {
    const struct stream_instruction *op = &batch->instructions[i];
    switch (op->operation) {
#include "instructions.inc"
    default:
      __builtin_trap();
    }
  }
  return 0;
}

#include <stdint.h>
struct instruction { uint64_t operation, rs1, rs2; };
struct batch { uint64_t count; struct instruction instructions[]; };
uint64_t ant_main(const struct batch *batch) {
  for (uint64_t i = 0; i < batch->count; ++i) {
    const struct instruction *op = &batch->instructions[i];
    switch (op->operation) {
#include "instructions.inc"
    default: __builtin_trap();
    }
  }
  return 0;
}

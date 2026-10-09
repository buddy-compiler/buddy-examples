#include "tlink.h"

#define COMMAND(op, field, value) ({ \
  uint64_t result; \
  __asm__ volatile(".insn r 0x2b, 7, " #op ", %0, %1, %2" \
    : "=r"(result) : "r"((uint64_t)(field)), "r"((uint64_t)(value)) : "memory"); \
  result; \
})

uint64_t tlink_query(enum tlink_field field) {
  return COMMAND(15, field, 0);
}

void tlink_transfer(uint64_t source_byte_address, uint32_t target_tile,
                    uint64_t target_byte_address, uint32_t bytes) {
  COMMAND(13, 0, source_byte_address);
  COMMAND(13, 1, target_tile);
  COMMAND(13, 2, target_byte_address);
  COMMAND(13, 3, bytes);
  if (COMMAND(14, 0, 0)) __builtin_trap();
}

uint64_t tlink_read64(uint64_t byte_address) {
  return COMMAND(16, byte_address, 0);
}

void tlink_write64(uint64_t byte_address, uint64_t data) {
  if (COMMAND(17, byte_address, data)) __builtin_trap();
}

uint64_t tlink_shared_export(uint32_t endpoint, uint32_t virtual_bank,
                            uint32_t group) {
  if (virtual_bank > UINT16_MAX || group > UINT16_MAX) __builtin_trap();
  uint64_t selector = ((uint64_t)endpoint << 32) |
                      ((uint64_t)virtual_bank << 16) | group;
  return COMMAND(18, selector, 0);
}

void tlink_shared_release(uint32_t endpoint, uint32_t virtual_bank,
                          uint32_t group) {
  if (virtual_bank > UINT16_MAX || group > UINT16_MAX) __builtin_trap();
  uint64_t selector = ((uint64_t)endpoint << 32) |
                      ((uint64_t)virtual_bank << 16) | group;
  if (COMMAND(19, selector, 0)) __builtin_trap();
}

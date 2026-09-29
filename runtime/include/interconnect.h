#pragma once
#include <stdint.h>

#define CHIP_LINK_BASE ((uintptr_t)0x62000000)
#define CHIP_LINK_BUFFER ((uintptr_t)0x62100000)
enum {
  LINK_CHIP_ID = 0,
  LINK_CHIP_COUNT = 8,
  LINK_SOURCE = 16,
  LINK_DEST_CHIP = 24,
  LINK_DESTINATION = 32,
  LINK_BYTES = 40,
  LINK_TAG = 48,
  LINK_SUBMIT = 56,
  LINK_STATUS = 64,
  LINK_EVENT_COUNT = 72,
  LINK_EVENT_SOURCE = 80,
  LINK_EVENT_TAG = 88,
  LINK_EVENT_BYTES = 96,
  LINK_EVENT_ACK = 104,
  LINK_MEMORY_BYTES = 112,
  LINK_EVENT_CAPACITY = 120,
  LINK_BUFFER_ADDRESS = 128,
  LINK_BUFFER_BYTES = 136
};
static inline uint64_t link_read(unsigned offset) {
  return *(volatile uint64_t *)(CHIP_LINK_BASE + offset);
}
static inline void link_write(unsigned offset, uint64_t value) {
  *(volatile uint64_t *)(CHIP_LINK_BASE + offset) = value;
}
static inline int chip_send(uint64_t target, uintptr_t source,
                            uintptr_t destination, uint64_t bytes,
                            uint64_t tag) {
  __asm__ volatile("fence rw,rw" ::: "memory");
  link_write(LINK_SOURCE, source);
  link_write(LINK_DEST_CHIP, target);
  link_write(LINK_DESTINATION, destination);
  link_write(LINK_BYTES, bytes);
  link_write(LINK_TAG, tag);
  link_write(LINK_SUBMIT, 1);
  uint64_t status;
  do {
    status = link_read(LINK_STATUS);
  } while (status == 1);
  __asm__ volatile("fence rw,rw" ::: "memory");
  return status == 2 ? 0 : -1;
}

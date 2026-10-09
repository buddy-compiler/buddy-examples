#include <dma.h>

#include <stdio.h>
#include <stdlib.h>

static _Thread_local uint32_t bank_cols[VIRTUAL_BANK_NUM];

void dma_bank_set_cols(uint32_t bank_id, uint32_t cols) {
  if (bank_id >= VIRTUAL_BANK_NUM) {
    fprintf(stderr, "dma_bank_set_cols: bank_id %u out of range\n", bank_id);
    exit(1);
  }
  bank_cols[bank_id] = cols;
}

void dma_bank_allocate(uint32_t bank_id, uint32_t cols) {
  uint32_t capacity =
      bank_id <= BB_PRIVATE_VBANK_MAX ? BANK_NUM : SHARED_BANK_NUM;
  dma_bank_set_cols(bank_id, cols ? cols : capacity);
}

uint32_t dma_bank_cols(uint32_t bank_id) {
  if (bank_id >= VIRTUAL_BANK_NUM) {
    fprintf(stderr, "dma_bank_cols: bank_id %u out of range\n", bank_id);
    exit(1);
  }
  return bank_cols[bank_id];
}

void dma_bank_transfer(uint64_t source, uint64_t target) {
  if (source >= VIRTUAL_BANK_NUM || target >= VIRTUAL_BANK_NUM) {
    fputs("dma_bank_transfer: bank id out of range\n", stderr);
    exit(1);
  }
  uint32_t columns = dma_bank_cols(source);
  uint32_t previous = dma_bank_cols(target);
  dma_bank_set_cols(target, previous + columns);
  dma_bank_set_cols(source, 0);
}

void dma_touch(void *p, size_t n) {
  volatile uint8_t *b = (volatile uint8_t *)p;
  size_t i;

  if (n == 0)
    return;
  for (i = 0; i < n; i += 4096)
    b[i] = b[i];
  b[n - 1] = b[n - 1];
}

void dma_touch_mvout(void *p, uint64_t depth, uint64_t stride,
                     uint32_t bank_id) {
  dma_touch(p, dma_span_bytes(depth, stride, dma_bank_cols(bank_id)));
}

void dma_touch_mvout_group(void *p, uint64_t depth, uint64_t stride) {
  dma_touch(p, dma_span_bytes(depth, stride, 1));
}

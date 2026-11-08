#include "dma_memory.h"
#include <stdint.h>

void dma_touch(void *p, size_t n) {
  volatile uint8_t *b = (volatile uint8_t *)p;
  if (n == 0)
    return;
  for (size_t i = 0; i < n; i += 4096)
    b[i] = b[i];
  b[n - 1] = b[n - 1];
}

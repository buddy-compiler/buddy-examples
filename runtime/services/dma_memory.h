#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Prepare private writable pages before an asynchronous DMA store.
void dma_touch(void *p, size_t n);

#ifdef __cplusplus
}
#endif

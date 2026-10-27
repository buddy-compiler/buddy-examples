#pragma once
#include "engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Ball instructions issued by host code run in batches on one hardware thread
 * whose program is the stream interpreter. */
struct stream_instruction {
  uint64_t operation, rs1, rs2;
};
struct stream {
  uint64_t count;
  struct stream_instruction instructions[];
};

/* Runs `entry` on this core; the Ball instructions it issues execute on a free
 * hardware thread of `signature` of engine_current(). Returns once that thread
 * has executed them all. */
void stream_run(uint64_t signature, void (*entry)(void *), void *argument);

/* The hardware thread executing the current stream_run, counted from 1; 0
 * outside. */
uint32_t stream_thread(void);

/* Called by lowered code, only inside stream_run. */
void stream_emit(uint32_t funct7, uint64_t rs1, uint64_t rs2);
void stream_flush(void);

#ifdef __cplusplus
}
#endif

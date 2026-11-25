#pragma once
#include "kernel.h"

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

/* Control words of the instruction stream; Ball funct7 occupies 1..127. */
enum { STREAM_LOOP = 128, STREAM_END, STREAM_OFFSET_1, STREAM_OFFSET_2 };
enum { STREAM_LOOP_DEPTH = 8, STREAM_STACK_BYTES = 1024 };
void stream_loop(uint64_t count);
void stream_end(void);
void stream_offset(uint64_t depth, uint64_t operand, uint64_t increment);
/* Retain a staging allocation until all jobs of the batch have finished. */
void stream_release(void *allocation);
/* Allocate aligned staging storage retained until the batch completes. */
void *stream_allocate(size_t bytes, size_t alignment);
/* Copy valid rows out of a padded result after the batch completes. */
void stream_copy(void *destination, const void *source, uint64_t rows,
                 uint64_t bytes, uint64_t destination_stride,
                 uint64_t source_stride);

/* Runs `entry` on this hart; the Ball instructions it issues execute on a free
 * hardware thread of `signature` of kernel_current(). Returns once that thread
 * has executed them all. */
void stream_run(uint64_t signature, void (*entry)(void *), void *argument);

/* Asynchronous jobs of compiled code on the hardware threads of
 * kernel_current(), one batch at a time per hart:
 *
 *   stream_begin(jobs, dependencies, transfers)
 *   stream_job(...) [stream_after(...)...]     once per job, in order
 *   stream_transfer(...)                        once per transfer
 *   stream_submit()                             the jobs start
 *   ... the hart keeps running ...
 *   stream_wait()                               every job has finished
 *
 * The counts are those of the whole batch. A job is the stream of Ball
 * instructions `entry` issues: stream_job runs `entry` on this hart right away
 * and records them, and the hardware thread executes the stream once the
 * batch is submitted and the job's dependencies and transfers are done. The
 * buffers a job reads and writes stay untouched by the hart until
 * stream_wait. Latency is counted on virtual clocks: stream_wait does not
 * count the hart's spinning and moves its mcycle to the end of the batch's
 * last job, each job lasting its thread's own cycles from when it could
 * start: the stream interpreter reads the thread's mcycle CSR at the start
 * and end of the stream, after a fence that waits for the thread's Balls to
 * finish, and returns the difference as the job's value
 * (machine mode: mcountinhibit, mcycle). */
void stream_begin(size_t jobs, size_t dependencies, size_t transfers);
/* Job `thread` (its index in the hart) of `signature`; jobs are numbered from
 * 0 in the order of the batch. An empty job records no instructions. */
void stream_job(uint64_t signature, uint32_t thread, void (*entry)(void *),
                void *argument, int empty);
/* The last job waits for the earlier job `dependency`. */
void stream_after(size_t dependency);
/* Job `target` reads, at `target_address`, the `bytes` job `source` writes at
 * `source_address`, passed between their threads by the hart
 * (kernel_transfer). */
void stream_transfer(size_t source, size_t target, void *source_address,
                     void *target_address, size_t bytes);
void stream_submit(void);
void stream_wait(void);

/* The hardware thread executing the current stream_run or recorded by the
 * current stream_job, counted from 1; 0 outside. */
uint32_t stream_thread(void);

/* Called by lowered code, only inside stream_run or stream_job; stream_flush
 * only inside stream_run. */
void stream_emit(uint32_t funct7, uint64_t rs1, uint64_t rs2);
/* Reserve consecutive instructions while recording a job. */
struct stream_instruction *stream_reserve(size_t count);
void stream_flush(void);

#ifdef __cplusplus
}
#endif

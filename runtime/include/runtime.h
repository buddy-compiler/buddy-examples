#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef void (*task_entry)(void *);
typedef struct task task;
typedef struct {
  uint32_t controller;
  size_t thread;
} thread_location_t;
thread_location_t thread_location(void);
void runtime_init(size_t stack_bytes);
void runtime_shutdown(void);
size_t thread_count(void);
uint64_t thread_signature(size_t thread);
size_t thread_submissions(size_t thread);
uint32_t thread_batches(size_t thread);
void workspace_init(void *memory, size_t bytes);
void workspace_begin(void *memory, size_t bytes);
size_t workspace_peak(void);
task *task_submit(uint64_t signature, task_entry entry, void *argument);
task *task_submit_on(size_t thread, uint64_t signature, task_entry entry,
                     void *argument);
int task_wait(task *task);
void task_run(uint64_t signature, task_entry entry, void *argument);
void task_run_on(size_t thread, uint64_t signature, task_entry entry,
                 void *argument);
void *workspace_alloc(size_t bytes);
void workspace_free(void *memory);

#ifdef __cplusplus
}
#endif

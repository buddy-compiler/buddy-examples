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
  size_t core;
} core_location_t;
core_location_t core_location(void);
void runtime_init(size_t stack_bytes);
void runtime_shutdown(void);
size_t core_count(void);
uint64_t core_signature(size_t core);
size_t core_submissions(size_t core);
void workspace_init(void *memory, size_t bytes);
void workspace_begin(void *memory, size_t bytes);
size_t workspace_peak(void);
task *task_submit(uint64_t signature, task_entry entry, void *argument);
task *task_submit_on(size_t core, uint64_t signature, task_entry entry,
                     void *argument);
int task_wait(task *task);
void task_run(uint64_t signature, task_entry entry, void *argument);
void task_run_on(size_t core, uint64_t signature, task_entry entry,
                 void *argument);
void *workspace_alloc(size_t bytes);
void workspace_free(void *memory);

#ifdef __cplusplus
}
#endif

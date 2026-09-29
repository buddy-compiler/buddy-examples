#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef void (*task_entry)(void *);
typedef struct task task;
void runtime_init(size_t stack_bytes);
size_t core_count(void);
uint64_t core_signature(size_t core);
void workspace_begin(void *memory, size_t bytes);
size_t workspace_peak(void);
task *task_submit(uint64_t signature, task_entry entry, void *argument);
int task_wait(task *task);
void task_run(uint64_t signature, task_entry entry, void *argument);
void *workspace_alloc(size_t bytes);
void workspace_free(void *memory);

#ifdef __cplusplus
}
#endif

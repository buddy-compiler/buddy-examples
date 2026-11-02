/* The device program runtime: a device program runs on the core of the
 * control hart as the first job of one batch and submits the jobs of the
 * cores while it runs (see program.c). A compiled program calls the job
 * functions; how a value moves between cores is the design's. */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Every hart calls this, once for each run of the program. The control hart
 * runs `program(core context, argument)` as the first job of the device and
 * returns its status once every job is done; the other harts run the jobs of
 * their cores and return 0. Under Linux one caller starts the workers and
 * returns after joining them. Runs on the same device are sequential. */
int program_run(int (*program)(void *, void *), void *argument);

/* The latency of the last run, from its start to the end of its last job, on
 * the virtual clocks of the cores, and the cycles the jobs of a core took;
 * once program_run returned. */
uint64_t program_latency(void);
uint64_t program_busy(uint64_t core);

/* The run the program is in, from 1: a job index names a job of its run
 * only. */
uint64_t program_run_index(void);

/* A job is begun on a core (an index in the device), given its words and the
 * jobs it waits for, and submitted; its entry receives the core context and
 * its words. Only the program submits jobs; the jobs of a core run in the
 * order they were submitted. */
void program_job_begin(uint64_t core, int (*entry)(void *, void *));
/* Begin a cleanup job after all submitted jobs. Append `count` malloc
 * pointers with program_job_argument, then submit it normally. */
void program_cleanup_begin(uint64_t count);
void program_job_argument(uint64_t word);
void program_job_after(uint64_t job);
uint64_t program_job_submit(void);
/* Returns once `job`, of a core other than the program's, has succeeded;
 * the program's clock moves to its end. */
void program_wait(uint64_t job);
/* Copies `bytes` from `source` to `target` on `core` once `after` is done;
 * returns the job of the copy. */
uint64_t program_copy(uint64_t core, uint64_t target, uint64_t source,
                      uint64_t bytes, uint64_t after);

#ifdef __cplusplus
}
#endif

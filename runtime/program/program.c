/* The device program runtime, on the scheduler (scheduler/runtime.h).
 *
 * A run is one batch of jobs on the device, and a program runs any number of
 * times, one run after another (a prefill, then decode steps): every hart
 * enters each run. The first job of a run is the program, on the
 * core of the control hart (DEVICE_CONTROL_HART); the program submits the
 * others while it runs, so the batch stays open until it returns. Every core
 * of the device (DEVICE_CORES of design.pb) has one worker, its hart, which
 * runs the jobs of its core in the order the program submitted them, the
 * program's core included once the program returned. The program may wait
 * for a job of another core, to read what it wrote.
 *
 * Latency is counted on a virtual clock per core, in the cycles of its hart
 * (mcycle), so that the same program gives the same latency however the
 * simulator interleaves the harts: a job starts when its core is free, the
 * program has submitted it and the jobs it waits for have ended, and lasts
 * the cycles its hart spends in it. The program is the first job of its
 * core: its clock advances by the cycles the program spends itself, and to
 * the end of a job it waits for.
 *
 * Every cycle whose count depends on the other harts is left out: the
 * spinning on the scheduler's lock (the program's submissions and waits are
 * not counted) and the allocator's critical section, spinning on its lock
 * included (its path depends on what the other cores allocated and freed).
 * Running the jobs of a core in submission order keeps the order, and so the
 * clocks, independent of when the other cores finish. */
#include "program.h"

#include <runtime.h>
#include <string.h>

#define PROGRAM_JOBS 8192
#define PROGRAM_WORDS 131072
#define PROGRAM_DEPENDENCIES 32768
/* The stack of every core's worker: the boot stack of a hart is 4 KiB, too
 * small for compiled graphs. */
#define PROGRAM_STACK_BYTES (1 << 20)

/* The cores of the device, in the order of design.pb. */
#define X(hart)                                                                \
  {DEVICE_CORES_##hart##_HART_ID, DEVICE_CORES_##hart##_SIGNATURE},
static const struct {
  uint32_t hart;
  uint64_t signature;
} device_cores[] = {DEVICE_CORES};
#undef X
#define CORE_COUNT (sizeof(device_cores) / sizeof(device_cores[0]))

static struct runtime_core cores[CORE_COUNT];
static struct runtime runtime;
static struct runtime_job jobs[PROGRAM_JOBS];
static struct runtime_result results[PROGRAM_JOBS];
static void *words[PROGRAM_WORDS];
static size_t word_count;
static size_t dependencies[PROGRAM_DEPENDENCIES];
static size_t dependency_count;
/* The job the program is building. */
static struct runtime_job job;
static uint32_t heap;
/* The runs the control hart started, the runs each hart entered, and the
 * harts' exits from runs: a run starts once every hart left the run before,
 * and its results are read once every hart left it. */
static uint32_t started;
static uint32_t entered_runs[CORE_COUNT];
static uint32_t left;

/* What each job runs, and when it may start and ended on the virtual clock
 * of its core. */
static struct timing {
  int (*entry)(void *, void *);
  void *words;
  size_t core;
  uint64_t ready, end;
  const size_t *dependencies;
  size_t dependency_count;
} timings[PROGRAM_JOBS];
static uint64_t clocks[CORE_COUNT];
static uint64_t busy[CORE_COUNT];
/* The cycles of each hart in the allocator, and its mcycle when it entered
 * it last. */
static uint64_t allocating[CORE_COUNT];
static uint64_t entered[CORE_COUNT];
/* The last job submitted to each core, plus one; 0 if none. */
static size_t last[CORE_COUNT];
static size_t control;
/* The control hart's mcycle when its clock last advanced. */
static uint64_t mark;
static int (*program_entry)(void *, void *);

static size_t core_of(uint32_t hart) {
  for (size_t core = 0; core < CORE_COUNT; ++core)
    if (device_cores[core].hart == hart)
      return core;
  __builtin_trap();
}

static size_t this_core(void) {
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  return core_of(hart);
}

/* The mcycle of the hart of `core` without its cycles in the allocator. */
static uint64_t cycles(size_t core) {
  uint64_t value;
  asm volatile("csrr %0, mcycle" : "=r"(value));
  return value - allocating[core];
}

/* The control core's clock, advanced by the cycles the program ran since
 * `mark`. */
static void advance(void) {
  const uint64_t now = cycles(control);
  clocks[control] += now - mark;
  busy[control] += now - mark;
  mark = now;
}

static int timed(void *context, void *argument) {
  struct timing *timing = argument;
  uint64_t begin = clocks[timing->core];
  if (timing->ready > begin)
    begin = timing->ready;
  for (size_t index = 0; index < timing->dependency_count; ++index)
    if (timings[timing->dependencies[index]].end > begin)
      begin = timings[timing->dependencies[index]].end;
  const uint64_t started = cycles(timing->core);
  const int status = timing->entry(context, timing->words);
  const uint64_t spent = cycles(timing->core) - started;
  timing->end = begin + spent;
  clocks[timing->core] = timing->end;
  busy[timing->core] += spent;
  return status;
}

static int run_program(void *context, void *argument) {
  mark = cycles(control);
  const int status = program_entry(context, argument);
  advance();
  return status;
}

/* The jobs of every core allocate their buffers: newlib's malloc is shared,
 * so it takes a lock. Its cycles from here to the unlock are not counted. */
void __malloc_lock(void *reent) {
  (void)reent;
  uint64_t now;
  asm volatile("csrr %0, mcycle" : "=r"(now));
  entered[this_core()] = now;
  while (__atomic_exchange_n(&heap, 1, __ATOMIC_ACQUIRE)) {
  }
}

void __malloc_unlock(void *reent) {
  (void)reent;
  const size_t core = this_core();
  uint64_t now;
  asm volatile("csrr %0, mcycle" : "=r"(now));
  allocating[core] += now - entered[core];
  __atomic_store_n(&heap, 0, __ATOMIC_RELEASE);
}

static uint8_t stacks[CORE_COUNT][PROGRAM_STACK_BYTES]
    __attribute__((aligned(16)));

static void work(size_t core) { runtime_baremetal_run(&runtime, core); }

int program_run(int (*program)(void *, void *), void *argument) {
  const size_t core = this_core();
  const uint32_t run = ++entered_runs[core];
  if (core == core_of(DEVICE_CONTROL_HART)) {
    while (__atomic_load_n(&left, __ATOMIC_ACQUIRE) != (run - 1) * CORE_COUNT) {
    }
    // A run of its own: its jobs, words and clocks from the start.
    word_count = 0;
    dependency_count = 0;
    memset(last, 0, sizeof(last));
    memset(clocks, 0, sizeof(clocks));
    memset(busy, 0, sizeof(busy));
    control = core;
    program_entry = program;
    for (size_t index = 0; index < CORE_COUNT; ++index)
      cores[index] = runtime_core(device_cores[index].hart,
                                  device_cores[index].signature, 0);
    runtime_init(&runtime, cores, CORE_COUNT, jobs, 0, results);
    const struct runtime_job first = {.signature = cores[control].signature,
                                      .core = control,
                                      .entry = run_program,
                                      .argument = argument};
    last[control] = runtime_submit(&runtime, &first) + 1;
    __atomic_store_n(&started, run, __ATOMIC_RELEASE);
  }
  while (__atomic_load_n(&started, __ATOMIC_ACQUIRE) < run) {
  }
  // The worker runs on the core's stack; the boot stack pointer is kept on
  // it and restored.
  register uintptr_t top asm("t1") = (uintptr_t)stacks[core + 1];
  register size_t worker_core asm("a0") = core;
  asm volatile("mv t0, sp\n"
               "addi sp, t1, -16\n"
               "sd t0, 0(sp)\n"
               "jalr %2\n"
               "ld sp, 0(sp)\n"
               : "+r"(worker_core), "+r"(top)
               : "r"(work)
               : "ra", "t0", "t2", "t3", "t4", "t5", "t6", "a1", "a2", "a3",
                 "a4", "a5", "a6", "a7", "ft0", "ft1", "ft2", "ft3", "ft4",
                 "ft5", "ft6", "ft7", "ft8", "ft9", "ft10", "ft11", "fa0",
                 "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7", "memory");
  __atomic_fetch_add(&left, 1, __ATOMIC_ACQ_REL);
  if (core != control)
    return 0;
  while (__atomic_load_n(&left, __ATOMIC_ACQUIRE) != run * CORE_COUNT) {
  }
  for (size_t index = 0; index < runtime.job_count; ++index)
    if (results[index].state != RUNTIME_SUCCEEDED || results[index].status)
      return index ? -1 - (int)index : results[0].status;
  return 0;
}

uint64_t program_latency(void) {
  uint64_t latency = 0;
  for (size_t core = 0; core < CORE_COUNT; ++core)
    if (clocks[core] > latency)
      latency = clocks[core];
  return latency;
}

uint64_t program_busy(uint64_t core) { return busy[core]; }

uint64_t program_run_index(void) { return started; }

/* The job the program submits next is the batch's next index: only the
 * program submits. */
void program_job_begin(uint64_t core, int (*entry)(void *, void *)) {
  if (runtime.job_count == PROGRAM_JOBS)
    __builtin_trap();
  timings[runtime.job_count] =
      (struct timing){.entry = entry,
                      .words = &words[word_count],
                      .core = core,
                      .dependencies = &dependencies[dependency_count]};
  job = (struct runtime_job){.signature = cores[core].signature,
                             .core = core,
                             .entry = timed,
                             .argument = &timings[runtime.job_count],
                             .dependencies = &dependencies[dependency_count],
                             .dependency_count = 0};
}

void program_job_argument(uint64_t word) {
  if (word_count == PROGRAM_WORDS)
    __builtin_trap();
  words[word_count++] = (void *)(uintptr_t)word;
}

void program_job_after(uint64_t after) {
  if (dependency_count == PROGRAM_DEPENDENCIES)
    __builtin_trap();
  dependencies[dependency_count++] = after;
  ++job.dependency_count;
}

uint64_t program_job_submit(void) {
  // After the last job of its core.
  if (last[job.core])
    program_job_after(last[job.core] - 1);
  advance();
  struct timing *timing = job.argument;
  timing->ready = clocks[control];
  timing->dependency_count = job.dependency_count;
  const size_t submitted = runtime_submit(&runtime, &job);
  last[job.core] = submitted + 1;
  mark = cycles(control);
  return submitted;
}

static int copy_entry(void *core, void *argument) {
  (void)core;
  void *const *copy = argument;
  memcpy(copy[0], copy[1], (uintptr_t)copy[2]);
  return 0;
}

uint64_t program_copy(uint64_t core, uint64_t target, uint64_t source,
                      uint64_t bytes, uint64_t after) {
  program_job_begin(core, copy_entry);
  program_job_argument(target);
  program_job_argument(source);
  program_job_argument(bytes);
  program_job_after(after);
  return program_job_submit();
}

void program_wait(uint64_t waited) {
  advance();
  if (timings[waited].core == control)
    __builtin_trap();
  for (;;) {
    while (__atomic_exchange_n(&runtime.lock, 1, __ATOMIC_ACQUIRE)) {
    }
    const enum runtime_state state = results[waited].state;
    __atomic_store_n(&runtime.lock, 0, __ATOMIC_RELEASE);
    if (state == RUNTIME_SUCCEEDED) {
      if (timings[waited].end > clocks[control])
        clocks[control] = timings[waited].end;
      mark = cycles(control);
      return;
    }
    if (state != RUNTIME_WAITING && state != RUNTIME_RUNNING)
      __builtin_trap();
  }
}

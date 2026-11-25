/* The executor of a static device program (program.h) for a device whose
 * harts share coherent DDR: a transfer is a copy at the front of the entry
 * that reads it first, its push nothing.
 *
 * Every hart of the device (DEVICE_HARTS of design.pb) has one worker, its
 * hart, which runs the entries of its hart in program order, following the
 * table's list of the hart. The program runs on the hart of the control hart
 * (DEVICE_CONTROL_HART) and dispatches the entries; that hart runs its own
 * entries while the program waits, in program_wait and program_finish. An
 * entry runs once it is dispatched and every wakeup it waits for has come:
 * its completion decrements the count of each entry it wakes and wakes the
 * hart of one that reaches zero. Entries retire in program order. A failed
 * entry squashes every later entry that has not started.
 *
 * Latency is counted on a virtual clock per hart, in the cycles of its hart
 * (mcycle): an entry starts when its hart is free, the program dispatched it
 * and the entries it waits for ended, and lasts the cycles its hart spends in
 * it; the program's hart advances by the cycles the program spends itself.
 * Waiting for the shared allocator's lock is left out. */
#ifdef PROGRAM_LINUX
#define _GNU_SOURCE
#endif
#include "program.h"

#include <runtime.h>
#include <stdlib.h>
#include <string.h>
#ifdef PROGRAM_LINUX
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#else
#include <baremetal.h>
#endif

/* The stack of every hart's worker: the boot stack of a hart is 4 KiB, too
 * small for compiled graphs. */
#define PROGRAM_STACK_BYTES (1 << 20)

/* The harts of the device, in the order of design.pb. */
#define X(hartid)                                                              \
  {DEVICE_HARTS_##hartid##_HART_ID, DEVICE_HARTS_##hartid##_SIGNATURE},
static const struct {
  uint32_t hartid;
  uint64_t signature;
} device_harts[] = {DEVICE_HARTS};
#undef X
#define HART_COUNT (sizeof(device_harts) / sizeof(device_harts[0]))

enum { WAITING, DONE, FAILED, SQUASHED };
/* The state of an entry. */
enum { PENDING, STATUS, AFTER, END, JOB };

static struct runtime_hart harts[HART_COUNT];
/* The run: its table, words and state, its counts. */
static const uint64_t *table;
static uint64_t *words, *state;
static uint64_t entry_count;
static const uint64_t *entries, *lists;
static const struct program_transfer *transfers;
static uint64_t dispatched, retired, failed;
/* The next entry of each hart, and the entry the program waits for. */
static uint64_t heads[HART_COUNT];
static uint64_t waited;
static size_t control;
static int (*program_entry)(void *, void *);
static void *program_argument;
static int program_status;
/* The runs the control hart started, each hart's runs, the harts' exits. */
static uint32_t started, left;
static uint32_t entered[HART_COUNT];
#ifdef PROGRAM_LINUX
/* The caller of program_run waits on it for the end of the run. */
static uint32_t over;
#endif

static uint64_t clocks[HART_COUNT], busy[HART_COUNT];
/* The program's clock mark: the cycles of its hart when it last advanced. */
static uint64_t mark;
#ifndef PROGRAM_LINUX
static uint64_t allocator_wait[HART_COUNT];
static unsigned allocation_depth[HART_COUNT];
static unsigned allocator_waiting[HART_COUNT];
static uint32_t heap;
#endif

static size_t hart_of(uint32_t hartid) {
  for (size_t hart = 0; hart < HART_COUNT; ++hart)
    if (device_harts[hart].hartid == hartid)
      return hart;
  __builtin_trap();
}

static uint64_t cycles(size_t hart) {
  uint64_t value;
#ifdef PROGRAM_LINUX
  (void)hart;
  asm volatile("rdcycle %0" : "=r"(value));
  return value;
#else
  asm volatile("csrr %0, mcycle" : "=r"(value));
  return value - allocator_wait[hart];
#endif
}

/* Waiting: a hart parks after `prepare` until woken; a wake between them is
 * not lost. */
#ifdef PROGRAM_LINUX
static uint32_t signals[HART_COUNT];
static uint32_t prepare(size_t hart) {
  return __atomic_load_n(&signals[hart], __ATOMIC_ACQUIRE);
}
static void park(size_t hart, uint32_t seen) {
  syscall(SYS_futex, &signals[hart], FUTEX_WAIT_PRIVATE, seen, NULL, NULL, 0);
}
static void wake(size_t hart) {
  __atomic_add_fetch(&signals[hart], 1, __ATOMIC_RELEASE);
  syscall(SYS_futex, &signals[hart], FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
}
#else
static uint32_t prepare(size_t hart) {
  runtime_baremetal_prepare_wait(device_harts[hart].hartid);
  return 0;
}
static void park(size_t hart, uint32_t seen) {
  (void)hart;
  (void)seen;
  asm volatile("wfi" ::: "memory");
}
static void wake(size_t hart) {
  runtime_baremetal_wake(device_harts[hart].hartid);
}
#endif

static const uint64_t *entry_of(uint64_t entry) {
  return &entries[entry * PROGRAM_ENTRY_WORDS];
}
static uint64_t *state_of(uint64_t entry) {
  return &state[entry * PROGRAM_STATE_WORDS];
}

static void raise_after(uint64_t entry, uint64_t stamp) {
  uint64_t *after = &state_of(entry)[AFTER];
  uint64_t seen = __atomic_load_n(after, __ATOMIC_RELAXED);
  while (seen < stamp &&
         !__atomic_compare_exchange_n(after, &seen, stamp, 1, __ATOMIC_ACQ_REL,
                                      __ATOMIC_RELAXED)) {
  }
}

/* The entries of `list` (an offset and a count in the entry) hear that an
 * entry ended at `stamp`. */
static void wakeups(const uint64_t *list, uint64_t stamp) {
  for (uint64_t index = 0; index < list[1]; ++index) {
    const uint64_t woken = lists[list[0] + index];
    raise_after(woken, stamp);
    if (!__atomic_sub_fetch(&state_of(woken)[PENDING], 1, __ATOMIC_ACQ_REL))
      wake(entry_of(woken)[0]);
  }
}

static void retire(void) {
  uint64_t oldest = __atomic_load_n(&retired, __ATOMIC_ACQUIRE);
  while (oldest < entry_count && __atomic_load_n(&state_of(oldest)[STATUS],
                                                 __ATOMIC_ACQUIRE) != WAITING) {
    if (__atomic_compare_exchange_n(&retired, &oldest, oldest + 1, 0,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
      ++oldest;
  }
  if (oldest == entry_count)
    wake(control);
}

static void finish_entry(uint64_t entry, uint64_t status) {
  __atomic_store_n(&state_of(entry)[STATUS], status, __ATOMIC_RELEASE);
  if (__atomic_load_n(&waited, __ATOMIC_ACQUIRE) == entry)
    wake(control);
  retire();
}

/* Runs the next entry of `hart` if it can: 1 if it ran or was squashed, 0 if
 * it waits, -1 if the hart has none left. */
static int step(size_t hart, void *context) {
  const uint64_t entry = heads[hart];
  if (entry >= entry_count)
    return -1;
  const uint64_t *fields = entry_of(entry);
  if (entry > __atomic_load_n(&failed, __ATOMIC_ACQUIRE)) {
    heads[hart] = fields[3];
    finish_entry(entry, SQUASHED);
    return 1;
  }
  if (entry >= __atomic_load_n(&dispatched, __ATOMIC_ACQUIRE) ||
      __atomic_load_n(&state_of(entry)[PENDING], __ATOMIC_ACQUIRE))
    return 0;
  heads[hart] = fields[3];
  uint64_t begin = clocks[hart];
  const uint64_t after =
      __atomic_load_n(&state_of(entry)[AFTER], __ATOMIC_ACQUIRE);
  if (after > begin)
    begin = after;
  const uint64_t first = cycles(hart);
  for (uint64_t index = 0; index < fields[9]; ++index) {
    const uint64_t *moved = &words[transfers[lists[fields[8] + index]].words];
    memcpy((void *)(uintptr_t)moved[1], (const void *)(uintptr_t)moved[0],
           moved[2]);
  }
  int (*job)(void *, void *) =
      (int (*)(void *, void *))(uintptr_t)state_of(entry)[JOB];
  const int status = job(context, &words[fields[1]]);
  const uint64_t spent = cycles(hart) - first;
  const uint64_t end = begin + spent;
  state_of(entry)[END] = end;
  clocks[hart] = end;
  busy[hart] += spent;
  if (status) {
    uint64_t seen = __atomic_load_n(&failed, __ATOMIC_ACQUIRE);
    while (entry < seen &&
           !__atomic_compare_exchange_n(&failed, &seen, entry, 0,
                                        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
    }
    // Every hart squashes its later entries.
    for (size_t other = 0; other < HART_COUNT; ++other)
      wake(other);
  }
  wakeups(&fields[4], end);
  wakeups(&fields[6], end);
  finish_entry(entry, status ? FAILED : DONE);
  return 1;
}

/* The program's clock, advanced by the cycles it ran since `mark`. */
static void advance(void) {
  const uint64_t now = cycles(control);
  clocks[control] += now - mark;
  busy[control] += now - mark;
  mark = now;
}

void program_start(const uint64_t *program, uint64_t *run_words,
                   uint64_t *run_state) {
  table = program;
  words = run_words;
  state = run_state;
  entry_count = table[0];
  const uint64_t hart_count = table[2];
  if (hart_count > HART_COUNT)
    __builtin_trap();
  entries = &table[3 + hart_count];
  transfers = (const struct program_transfer
                   *)&entries[entry_count * PROGRAM_ENTRY_WORDS];
  lists = (const uint64_t *)&transfers[table[1]];
  for (size_t hart = 0; hart < HART_COUNT; ++hart)
    heads[hart] = hart < hart_count ? table[3 + hart] : entry_count;
  for (uint64_t entry = 0; entry < entry_count; ++entry) {
    uint64_t *fields = state_of(entry);
    fields[PENDING] = entry_of(entry)[2];
    fields[STATUS] = WAITING;
    fields[AFTER] = 0;
    fields[END] = 0;
  }
  dispatched = 0;
  retired = 0;
  failed = entry_count;
  waited = entry_count;
  mark = cycles(control);
  __atomic_store_n(&started, started + 1, __ATOMIC_RELEASE);
  for (size_t hart = 0; hart < HART_COUNT; ++hart)
    if (hart != control)
      wake(hart);
}

void program_dispatch(uint64_t entry, int (*job)(void *, void *)) {
  advance();
  if (entry != dispatched)
    __builtin_trap();
  state_of(entry)[JOB] = (uintptr_t)job;
  raise_after(entry, clocks[control]);
  __atomic_store_n(&dispatched, entry + 1, __ATOMIC_RELEASE);
  if (!__atomic_load_n(&state_of(entry)[PENDING], __ATOMIC_ACQUIRE))
    wake(entry_of(entry)[0]);
  mark = cycles(control);
}

/* The program's hart runs its own entries until `done`. */
static void serve(void *context, int (*done)(void)) {
  for (;;) {
    const uint32_t seen = prepare(control);
    if (done())
      return;
    if (step(control, context) <= 0)
      park(control, seen);
  }
}

static void *program_context;

static int waited_done(void) {
  return __atomic_load_n(&state_of(waited)[STATUS], __ATOMIC_ACQUIRE) !=
         WAITING;
}

void program_wait(uint64_t entry) {
  advance();
  __atomic_store_n(&waited, entry, __ATOMIC_RELEASE);
  serve(program_context, waited_done);
  if (__atomic_load_n(&state_of(entry)[STATUS], __ATOMIC_ACQUIRE) != DONE)
    __builtin_trap();
  waited = entry_count;
  if (state_of(entry)[END] > clocks[control])
    clocks[control] = state_of(entry)[END];
  mark = cycles(control);
}

static int all_retired(void) {
  return __atomic_load_n(&retired, __ATOMIC_ACQUIRE) == entry_count;
}

void program_finish(void) {
  advance();
  serve(program_context, all_retired);
}

int program_copy_job(void *hart, void *argument) {
  (void)hart;
  void *const *copy = argument;
  memcpy(copy[0], copy[1], (uintptr_t)copy[2]);
  return 0;
}

uint64_t program_latency(void) {
  uint64_t latency = 0;
  for (size_t hart = 0; hart < HART_COUNT; ++hart)
    if (clocks[hart] > latency)
      latency = clocks[hart];
  return latency;
}

uint64_t program_busy(uint64_t hart) { return busy[hart]; }

/* A worker: the program on the control hart, the entries elsewhere, for one
 * run. */
static void work_run(size_t hart, void *context, uint32_t run) {
  if (hart == control) {
    clocks[hart] = busy[hart] = 0;
    program_context = context;
    program_status = program_entry(context, program_argument);
    advance();
  } else {
    for (;;) {
      const uint32_t seen = prepare(hart);
      if (__atomic_load_n(&started, __ATOMIC_ACQUIRE) == run)
        break;
      park(hart, seen);
    }
    clocks[hart] = busy[hart] = 0;
    for (;;) {
      const uint32_t seen = prepare(hart);
      const int progress = step(hart, context);
      if (progress < 0)
        break;
      if (!progress)
        park(hart, seen);
    }
  }
#ifdef PROGRAM_LINUX
  if (__atomic_add_fetch(&left, 1, __ATOMIC_ACQ_REL) == HART_COUNT) {
    __atomic_add_fetch(&over, 1, __ATOMIC_RELEASE);
    syscall(SYS_futex, &over, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
  }
#else
  __atomic_fetch_add(&left, 1, __ATOMIC_ACQ_REL);
#endif
}

static int result(void) {
  if (program_status)
    return program_status;
  return failed < entry_count ? -1 - (int)failed : 0;
}

#ifdef PROGRAM_LINUX
static pthread_t threads[HART_COUNT];
/* The run the workers are asked for. */
static uint32_t requested;

static int work(void *context, void *argument) {
  const size_t hart = (uintptr_t)argument;
  for (uint32_t run = 1;; ++run) {
    for (;;) {
      const uint32_t seen = prepare(hart);
      if (__atomic_load_n(&requested, __ATOMIC_ACQUIRE) >= run)
        break;
      park(hart, seen);
    }
    work_run(hart, context, run);
  }
  return 0;
}

static void *thread(void *argument) {
  const size_t hart = (uintptr_t)argument;
  if (harts[hart].execute(harts[hart].context, work, argument))
    abort();
  return NULL;
}

int program_run(int (*program)(void *, void *), void *argument) {
  control = hart_of(DEVICE_CONTROL_HART);
  program_entry = program;
  program_argument = argument;
  const uint32_t run = requested + 1;
  if (run == 1) {
    unsigned cpus[HART_COUNT];
    for (size_t index = 0; index < HART_COUNT; ++index)
      cpus[index] = UINT32_MAX;
    FILE *info = fopen("/proc/cpuinfo", "r");
    if (!info)
      abort();
    char line[256];
    unsigned cpu = UINT32_MAX, hartid;
    while (fgets(line, sizeof(line), info)) {
      if (sscanf(line, "processor : %u", &cpu) == 1)
        continue;
      if (sscanf(line, "hart : %u", &hartid) == 1)
        for (size_t index = 0; index < HART_COUNT; ++index)
          if (device_harts[index].hartid == hartid)
            cpus[index] = cpu;
    }
    fclose(info);
    pthread_attr_t attributes;
    if (pthread_attr_init(&attributes) ||
        pthread_attr_setstacksize(&attributes, PROGRAM_STACK_BYTES))
      abort();
    for (size_t hart = 0; hart < HART_COUNT; ++hart) {
      if (cpus[hart] == UINT32_MAX)
        abort();
      harts[hart] = runtime_hart(device_harts[hart].hartid,
                                 device_harts[hart].signature, 0);
      cpu_set_t affinity;
      CPU_ZERO(&affinity);
      CPU_SET(cpus[hart], &affinity);
      if (pthread_attr_setaffinity_np(&attributes, sizeof(affinity),
                                      &affinity) ||
          pthread_create(&threads[hart], &attributes, thread,
                         (void *)(uintptr_t)hart))
        abort();
    }
    pthread_attr_destroy(&attributes);
  }
  left = 0;
  __atomic_store_n(&requested, run, __ATOMIC_RELEASE);
  for (size_t hart = 0; hart < HART_COUNT; ++hart)
    wake(hart);
  // The caller is no worker: it waits on a word of its own.
  for (;;) {
    const uint32_t seen = __atomic_load_n(&over, __ATOMIC_ACQUIRE);
    if (__atomic_load_n(&left, __ATOMIC_ACQUIRE) == HART_COUNT)
      break;
    syscall(SYS_futex, &over, FUTEX_WAIT_PRIVATE, seen, NULL, NULL, 0);
  }
  return result();
}
#else
/* Newlib's shared heap needs a recursive lock. Only acquisition waiting is
 * excluded from hart-local timing; allocator execution remains counted. */
static size_t this_hart(void) {
  uintptr_t hartid;
  asm volatile("csrr %0, mhartid" : "=r"(hartid));
  return hart_of(hartid);
}

void __malloc_lock(void *reent) {
  (void)reent;
  const size_t hart = this_hart();
  if (allocation_depth[hart]++)
    return;
  uint64_t begin, acquired;
  asm volatile("csrr %0, mcycle" : "=r"(begin));
  __atomic_store_n(&allocator_waiting[hart], 1, __ATOMIC_RELEASE);
  for (;;) {
    runtime_baremetal_prepare_wait(device_harts[hart].hartid);
    if (!__atomic_exchange_n(&heap, 1, __ATOMIC_ACQUIRE))
      break;
    asm volatile("wfi" ::: "memory");
  }
  __atomic_store_n(&allocator_waiting[hart], 0, __ATOMIC_RELEASE);
  asm volatile("csrr %0, mcycle" : "=r"(acquired));
  allocator_wait[hart] += acquired - begin;
}

void __malloc_unlock(void *reent) {
  (void)reent;
  const size_t hart = this_hart();
  if (--allocation_depth[hart])
    return;
  __atomic_store_n(&heap, 0, __ATOMIC_RELEASE);
  for (size_t waiting = 0; waiting < HART_COUNT; ++waiting)
    if (__atomic_load_n(&allocator_waiting[waiting], __ATOMIC_ACQUIRE))
      runtime_baremetal_wake(device_harts[waiting].hartid);
}

static uint8_t stacks[HART_COUNT][PROGRAM_STACK_BYTES]
    __attribute__((aligned(16)));

struct worker {
  size_t hart;
  uint32_t run;
};

static int work(void *context, void *argument) {
  const struct worker *worker = argument;
  work_run(worker->hart, context, worker->run);
  return 0;
}

static void run_on_stack(size_t hart, uint32_t run) {
  struct worker worker = {hart, run};
  harts[hart] =
      runtime_hart(device_harts[hart].hartid, device_harts[hart].signature, 0);
  if (harts[hart].execute(harts[hart].context, work, &worker))
    __builtin_trap();
}

int program_run(int (*program)(void *, void *), void *argument) {
  const size_t hart = this_hart();
  const uint32_t run = ++entered[hart];
  control = hart_of(DEVICE_CONTROL_HART);
  if (hart == control) {
    // A run starts once every hart left the run before.
    while (__atomic_load_n(&left, __ATOMIC_ACQUIRE) != (run - 1) * HART_COUNT) {
    }
    program_entry = program;
    program_argument = argument;
  }
  // The worker runs on the hart's stack; the boot stack pointer is kept on
  // it and restored.
  register uintptr_t top asm("t1") = (uintptr_t)stacks[hart + 1];
  register size_t worker_hart asm("a0") = hart;
  register uint32_t worker_run asm("a1") = run;
  asm volatile("mv t0, sp\n"
               "addi sp, t1, -16\n"
               "sd t0, 0(sp)\n"
               "jalr %3\n"
               "ld sp, 0(sp)\n"
               : "+r"(worker_hart), "+r"(top), "+r"(worker_run)
               : "r"(run_on_stack)
               : "ra", "t0", "t2", "t3", "t4", "t5", "t6", "a2", "a3", "a4",
                 "a5", "a6", "a7", "ft0", "ft1", "ft2", "ft3", "ft4", "ft5",
                 "ft6", "ft7", "ft8", "ft9", "ft10", "ft11", "fa0", "fa1",
                 "fa2", "fa3", "fa4", "fa5", "fa6", "fa7", "memory");
  if (hart != control)
    return 0;
  while (__atomic_load_n(&left, __ATOMIC_ACQUIRE) != run * HART_COUNT) {
  }
  return result();
}
#endif

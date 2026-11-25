/* The device program runtime: a device program is static, lowered by
 * lower-device (stack/compiler/midend/Conversion/LowerDevice.cpp). Its
 * entries (a launch of a graph on a hart, or a copy into a result), in
 * program order, what each waits for and wakes, and the transfers of values
 * between them are fixed in its table; the program fills in the words of the
 * run (addresses and sizes) and dispatches the entries in order. Each hart
 * runs its entries in program order, one at a time, an entry once every entry
 * it waits for woke it; entries retire in program order. An entry that fails
 * stops the run precisely: the entries before it take effect, the entries
 * after it that have not started are squashed. How a transfer moves its value
 * belongs to the executor: a transfer is pulled at the front of the entry
 * that reads it first and pushed at the tail of the entry that wrote it.
 *
 * This is the contract of a lowered program. The executor of a design that
 * moves values over coherent DDR is program.c here (a pull is a copy, a push
 * nothing); a design with its own transfer hardware implements the contract
 * in its device runtime. */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The words of an entry in the table: its hart, its first word, the number
 * of wakeups it waits for, the next entry of its hart, and an offset and a
 * count into the lists for each of: the entries its completion wakes, the
 * entries its pushes wake (stream wakeups), the transfers it pulls, the
 * transfers it pushes. */
#define PROGRAM_ENTRY_WORDS 12
/* The words of the state of an entry, zero before the first run. */
#define PROGRAM_STATE_WORDS 5

/* A transfer in the table; its words are its source, target, byte count and
 * the compiler's logical value identity. */
struct program_transfer {
  uint64_t from, to, words;
};

/* Every hart calls this, once for each run of the program. The control hart
 * (DEVICE_CONTROL_HART) runs `program(hart context, argument)`, which starts
 * the run, and returns the program's status, or the failure of the first
 * entry that failed (-1 - entry); the other harts run the entries of their
 * harts and return 0. Under Linux one caller runs the workers of every hart
 * and returns once the run is over. Runs are sequential. */
int program_run(int (*program)(void *, void *), void *argument);

/* The latency of the last run on the virtual clocks of the harts (in the
 * cycles of their harts, so that it does not depend on how a simulator
 * interleaves them), and the cycles each hart spent in its entries; once
 * program_run returned. */
uint64_t program_latency(void);
uint64_t program_busy(uint64_t hart);

/* The calls of a lowered program (lower-device). */
void program_start(const uint64_t *table, uint64_t *words, uint64_t *state);
void program_dispatch(uint64_t entry, int (*job)(void *, void *));
/* Returns once `entry` has completed; the program's hart runs its own
 * entries meanwhile. */
void program_wait(uint64_t entry);
/* Returns once every entry has retired. */
void program_finish(void);
/* The job of a copy entry: its words are target, source, bytes. */
int program_copy_job(void *hart, void *words);

#ifdef __cplusplus
}
#endif

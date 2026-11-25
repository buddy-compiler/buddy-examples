# Kernel

Kernel schedules OS-invisible hardware threads within one hart. It runs on the
hart's CPU hart, uses the Ant management ISA, and creates no OS threads; only the
stream batches below allocate memory. The runtime selects the hart; kernel selects a compatible local
hardware thread. Keep every call in one kernel session on the same hart and in the
same address space.

The caller provides an `kernel`, a slot array, a job array, and a result array.
`kernel_init` queries hardware capabilities. `kernel_begin` takes group ownership
and starts ready jobs; `kernel_publish` adds the next jobs of the same array to
a batch without transfers. `kernel_poll` checks every running thread,
acknowledges completed work, and reuses available threads, each thread running
its jobs in their order. Its return value is the number of
unfinished jobs. `kernel_wait` polls until that count is zero; `kernel_end` then
releases group ownership. `kernel_run` performs that sequence and returns the
number of cancelled jobs.

A job selects an explicit thread or `KERNEL_ANY_THREAD`, names its program
signature, and can depend on earlier job indices. The scheduler requires matching
signatures. A cancelled dependency cancels its waiting dependents. The binding's `load(thread)`
loads the thread's program once, at `kernel_init`; the per-job
`prepare(thread, opaque)` callback loads the job's arguments into its TLS;
TSS must be initialized before `kernel_begin`. All job, dependency, callback, and
result storage remains valid through `kernel_end`. Scheduling is cooperative:
call `kernel_poll` to make progress while doing CPU work.

`kernel_cancel` requests cancellation of one job. `kernel_cancel_all` requests it
for the entire batch. Running work stays in `KERNEL_CANCELLING` until hardware
publishes completion after draining accepted operations. A racing normal
completion may still succeed. Cancellation does not release the thread or group;
continue polling, then call `kernel_end`. Invalid descriptors, unsupported
signatures, invalid dependencies, and premature release are programming errors.

Compiled code runs its hardware-thread work through `stream.h`. `stream_run`
runs a function on the CPU and executes the Ball instructions it issues on a
thread, synchronously. A batch is asynchronous: `stream_begin` opens it with its
job, dependency and transfer counts, `stream_job` records the instructions a
function issues as the next job for an explicit thread, `stream_after` and
`stream_transfer` add its dependencies and the compiled thread-to-thread
transfers, `stream_submit` gives the kernel the rest of the batch, and
`stream_wait` waits for it and releases it. A batch without transfers is
given to the kernel job by job: recording a job starts the batch
(`kernel_begin`) or publishes the jobs before it (`kernel_publish`), so the
threads run them while the hart records the rest; a batch with transfers is
given whole at `stream_submit`. The batch allocates its arrays and
recorded streams with `malloc` and frees them in `stream_wait`; a recorded
stream must fit the thread's TLS and TSS. Latency is counted on virtual clocks,
like the device program runtime does across harts: each thread runs its jobs in
batch order, a job starts once the hart has given it to the kernel (the hart's
`mcycle`), its thread is free and its dependencies and transfer sources have
ended, and lasts
the thread's own cycles in it, which the stream interpreter measures with the
thread's `mcycle` CSR (read at the start and the end of the stream) and returns
as the job's value; `stream_wait` does not count the hart's spinning
(`mcountinhibit`) and moves its `mcycle` to the end of the last job, in
machine mode; under Linux the clocks are the OS's and `stream_wait` only
waits. The interpreter fences before its last mcycle read: a fence on the
Ant waits for its accelerator and local stores to drain, so the duration covers
the Balls' work after the last instruction issued.

`kernel_init` loads the program of every online thread once (`load` of its
binding); a job only hands the thread its arguments (`prepare`).

On any hart with hardware threads the kernel is online, baremetal and under
Linux alike, with the same API. Baremetal, the kernel issues the hart's
management instructions itself (`src/thread.c`). Under Linux the OS owns the
threads: the kernel hands the same instructions to the kernel driver
(`stack/os/linux/driver`, `/dev/kernel`, `src/thread_linux.c`), which issues
them in order in the kernel on the CPU the calling thread is pinned to, with
SUM set so that a task started there reaches the process's pages; the first
list on a CPU makes the open file the owner of that hart's threads, and
closing it cancels and releases what is left. A Linux program reaches a hart's
kernel through a runtime job on that hart (`runtime_linux_job`).

The `kernel` CMake target contains the baremetal CPU-side library and inherits
the caller's ISA/ABI; `kernel-linux` (static Linux programs) and
`kernel-shared` (shared objects, such as models) are the Linux ones. The separate `ant_stream` target builds an
RV64IM command-stream program; a concrete design chooses its program image and
signature rather than assuming that image fits every thread ISA.

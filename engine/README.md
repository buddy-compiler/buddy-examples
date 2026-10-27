# Buckyball Engine

Engine schedules OS-invisible hardware threads within one core. It runs on the
core's CPU hart, uses the Ant management ISA, and creates no OS threads or dynamic
allocations. The runtime selects the core; engine selects a compatible local
hardware thread. Keep every call in one engine session on the same hart and in the
same address space.

The caller provides an `engine`, a slot array, a job array, and a result array.
`engine_init` queries hardware capabilities. `engine_begin` takes group ownership
and starts ready jobs. `engine_poll` checks every running thread, acknowledges
completed work, and reuses available threads. Its return value is the number of
unfinished jobs. `engine_wait` polls until that count is zero; `engine_end` then
releases group ownership. `engine_run` performs that sequence and returns the
number of cancelled jobs.

A job selects an explicit thread or `ENGINE_ANY_THREAD`, names its program
signature, and can depend on earlier job indices. The scheduler requires matching
signatures. A cancelled dependency cancels its waiting dependents. The required per-job
`prepare(thread, opaque)` callback loads the selected thread's code and TLS
(or does nothing when the program is already loaded);
TSS must be initialized before `engine_begin`. All job, dependency, callback, and
result storage remains valid through `engine_end`. Scheduling is cooperative:
call `engine_poll` to make progress while doing CPU work.

`engine_cancel` requests cancellation of one job. `engine_cancel_all` requests it
for the entire batch. Running work stays in `ENGINE_CANCELLING` until hardware
publishes completion after draining accepted operations. A racing normal
completion may still succeed. Cancellation does not release the thread or group;
continue polling, then call `engine_end`. Invalid descriptors, unsupported
signatures, invalid dependencies, and premature release are programming errors.

The `engine` CMake target contains the CPU-side library and inherits the caller's
ISA/ABI. The separate `ant_stream` target builds an
RV64IM command-stream program; a concrete design chooses its program image and
signature rather than assuming that image fits every thread ISA.

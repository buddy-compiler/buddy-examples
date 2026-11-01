#include "stream.h"

/* Trace queries must also link on cores without hardware threads. */
_Thread_local uint32_t stream_target = UINT32_MAX;

uint32_t stream_thread(void) { return stream_target + 1; }

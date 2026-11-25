#pragma once

#include "runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Clear before checking the condition: a later notification must remain
 * pending until WFI observes it. Global machine interrupts remain disabled. */
void runtime_baremetal_prepare_wait(uint32_t hartid);
void runtime_baremetal_wake(uint32_t hartid);
/* Wakes the harts of `harts`, a bit per index. */
void runtime_baremetal_notify(const struct runtime *runtime, uint64_t harts);

#ifdef __cplusplus
}
#endif

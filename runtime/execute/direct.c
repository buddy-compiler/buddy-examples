#include "runtime.h"

/* Execution for a hart without hardware threads: the job runs on the hart
 * itself. */
static int execute(void *context, int (*entry)(void *, void *),
                   void *argument) {
  return entry(context, argument);
}

static const struct runtime_binding binding
    __attribute__((section("runtime_bindings"), used)) = {HART_SIGNATURE,
                                                          execute};

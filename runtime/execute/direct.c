#include "runtime.h"

/* Execution for a core without hardware threads: the job runs on the core
 * itself. */
static int execute(void *context, int (*entry)(void *, void *),
                   void *argument) {
  return entry(context, argument);
}

static const struct runtime_binding binding
    __attribute__((section("runtime_bindings"), used)) = {CORE_SIGNATURE,
                                                          execute};

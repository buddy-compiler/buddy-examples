#include "runtime.h"

void runtime_baremetal_run(struct runtime *runtime, size_t core) {
  while (runtime_step(runtime, core) != RUNTIME_DONE) {
  }
}

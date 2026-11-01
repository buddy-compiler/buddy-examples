#include <program.h>

extern "C" int program_job(void *, void *);

extern "C" int gemma_run(void **words) {
  return program_run(program_job, words);
}

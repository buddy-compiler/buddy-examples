#include <program.h>
extern "C" int program_job(void *, void *);
extern "C" int smolvla_run(void **arguments) {
  return program_run(program_job, arguments);
}

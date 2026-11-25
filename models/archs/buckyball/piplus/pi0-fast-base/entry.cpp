#include <program.h>
extern "C" int prefill_program_job(void *, void *);
extern "C" int decode_program_job(void *, void *);
extern "C" int pi0_fast_prefill(void **arguments) {
  return program_run(prefill_program_job, arguments);
}
extern "C" int pi0_fast_decode(void **arguments) {
  return program_run(decode_program_job, arguments);
}

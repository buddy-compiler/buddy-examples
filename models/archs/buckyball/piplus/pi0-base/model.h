#pragma once
#include <stdint.h>

struct Observation {
  float images[3 * 3 * 224 * 224];
  int64_t tokens[48], masks[48];
  float state[32], noise[ACTION_ELEMENTS];
};
static_assert(sizeof(Observation) == 1806336 + 768 + 128 + 4 * ACTION_ELEMENTS);

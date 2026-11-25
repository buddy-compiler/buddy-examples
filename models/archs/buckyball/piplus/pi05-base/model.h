#pragma once
#include <stdint.h>

struct Observation {
  float images[3 * 3 * 224 * 224];
  int64_t tokens[200], masks[200];
  float noise[ACTION_ELEMENTS];
};
static_assert(sizeof(Observation) ==
              1806336 + 1600 + 1600 + 4 * ACTION_ELEMENTS);

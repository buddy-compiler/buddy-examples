#pragma once
#include <stdint.h>
struct Input {
  int64_t tokens[128], mask[128], positions[16];
  bool valid[16];
  int64_t type[1];
};
static_assert(sizeof(Input) == 2200);

#pragma once
#include <stdint.h>

struct Observation {
  float images[3 * 3 * 224 * 224];
  // The prompt, then the token the model continues.
  int64_t tokens[201], masks[201];
};
static_assert(sizeof(Observation) == 1806336 + 1608 + 1608);

// The Gemma-2B of PaliGemma as the graphs of the recipe take it
// (frontend/graphs.py): its layers, the keys of the prefill (padded), the
// tokens a run generates at most, the keys a decoding step brings itself,
// the values of a key and the vocabulary; the EOS of the vocabulary.
#define LAYERS 18
#define PREFIX_KEYS 992
#define GENERATED 256
#define OWN_KEYS 32
#define HEAD_DIM 256
#define VOCABULARY 257152
#define HIDDEN 2048
#define EOS 1

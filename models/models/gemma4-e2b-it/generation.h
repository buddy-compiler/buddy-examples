#pragma once
#include "gemma-parameters.h"
#include <cstddef>
#include <cstdint>
#include <vector>

inline constexpr size_t MaxVocabSize = 262144;

class Execution;
std::vector<int64_t> generate(Execution &execution,
                              const std::vector<int64_t> &tokens,
                              const std::vector<int64_t> &eos,
                              size_t maxTokens);

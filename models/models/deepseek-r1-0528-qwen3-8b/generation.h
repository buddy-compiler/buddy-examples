#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
class Execution;
std::vector<int64_t> generate(Execution &execution,
                              std::vector<int64_t> request,
                              const std::vector<int64_t> &eos, size_t maxTokens,
                              double temperature);

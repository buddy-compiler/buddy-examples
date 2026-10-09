#include "generation.h"
#include "execution.h"
#include <algorithm>

std::vector<int64_t> generate(Execution &execution,
                              const std::vector<int64_t> &tokens,
                              const std::vector<int64_t> &eos,
                              size_t maxTokens) {
  execution.prefill(tokens);
  auto choose = [&]() {
    auto logits = execution.logits();
    return int64_t(std::max_element(logits.begin(), logits.end()) -
                   logits.begin());
  };
  int64_t token = choose();
  std::vector<int64_t> generated{token};
  for (size_t step = 1; step < maxTokens; ++step) {
    execution.decode(token);
    token = choose();
    if (std::find(eos.begin(), eos.end(), token) != eos.end())
      break;
    generated.push_back(token);
  }
  return generated;
}

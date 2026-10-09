#include "generation.h"
#include "execution.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <random>
#include <stdexcept>

std::vector<int64_t> generate(Execution &execution,
                              std::vector<int64_t> request,
                              const std::vector<int64_t> &eos, size_t maxTokens,
                              double temperature) {
  std::vector<int64_t> generated;
  std::optional<std::mt19937_64> random;
  if (temperature > 0)
    random.emplace(std::random_device{}());
  execution.prefill(request);
  for (size_t step = 0; step < maxTokens; ++step) {
    const auto logits = execution.logits();
    for (float value : logits)
      if (!std::isfinite(value))
        throw std::runtime_error("model produced non-finite logits");
    int64_t token;
    if (temperature == 0) {
      token = std::max_element(logits.begin(), logits.end()) - logits.begin();
    } else {
      const double maximum = *std::max_element(logits.begin(), logits.end());
      std::vector<double> probabilities(logits.size());
      for (size_t index = 0; index < logits.size(); ++index)
        probabilities[index] =
            std::exp((double(logits[index]) - maximum) / temperature);
      std::discrete_distribution<size_t> sample(probabilities.begin(),
                                                probabilities.end());
      token = sample(*random);
    }
    generated.push_back(token);
    if (std::find(eos.begin(), eos.end(), token) != eos.end())
      break;
    if (step + 1 < maxTokens)
      execution.decode(token);
  }
  return generated;
}

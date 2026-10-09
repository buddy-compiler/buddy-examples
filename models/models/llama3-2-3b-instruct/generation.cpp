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
  const auto &shape = modelShape();
  std::vector<float> reduced(shape.prefill * shape.hidden);
  std::vector<float> logits(shape.vocabulary);
  std::vector<int64_t> generated;
  std::optional<std::mt19937_64> random;
  if (temperature > 0)
    random.emplace(std::random_device{}());
  size_t start = 0;
  for (size_t step = 0; step < maxTokens; ++step) {
    const size_t count = request.size();
    const size_t length = start == 0 ? shape.prefill : 1;
    std::fill_n(execution.tokens, shape.prefill, int64_t(0));
    std::copy(request.begin(), request.end(), execution.tokens);
    execution.submit(0, 0, count, start, 0);
    execution.wait(0);
    std::memcpy(execution.hidden, execution.workers[0].output,
                count * shape.hidden * sizeof(float));
    for (size_t layer = 0; layer < shape.layers; ++layer) {
      for (uint64_t operation : {2, 3}) {
        for (size_t rank = 0; rank < shape.tiles; ++rank) {
          auto &worker = execution.workers[rank];
          std::fill_n(worker.input, length * shape.hidden, 0.0f);
          std::memcpy(worker.input, execution.hidden,
                      count * shape.hidden * sizeof(float));
          execution.submit(rank, operation, count, start, layer);
        }
        std::fill_n(reduced.data(), count * shape.hidden, 0.0f);
        for (size_t rank = 0; rank < shape.tiles; ++rank) {
          execution.wait(rank);
          for (size_t index = 0; index < count * shape.hidden; ++index)
            reduced[index] += execution.workers[rank].output[index];
        }
        for (size_t index = 0; index < count * shape.hidden; ++index)
          execution.hidden[index] += reduced[index];
      }
    }
    std::memcpy(execution.workers[0].input,
                execution.hidden + (count - 1) * shape.hidden,
                shape.hidden * sizeof(float));
    execution.submit(0, 1, 1, 0, 0);
    execution.wait(0);
    std::memcpy(logits.data(), execution.workers[0].output,
                shape.vocabulary * sizeof(float));
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
    start += count;
    if (std::find(eos.begin(), eos.end(), token) != eos.end())
      break;
    request.assign(1, token);
  }
  return generated;
}

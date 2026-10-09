#include "model.h"
#include <cmath>
#include <iomanip>
#include <iostream>

void decision(Context &ctx, size_t options, size_t actions,
              double temperature) {
  size_t count = 0;
  for (size_t index = 0; index < options; ++index)
    count += ctx.valid.getData()[index] != 0;
  if (!count || !std::isfinite(temperature))
    throw std::runtime_error("invalid Laya calibration input");
  temperature = std::clamp(temperature, 0.5, 5.0);
  std::vector<double> probabilities(count), actionProbabilities(actions);
  double maximum = -INFINITY;
  for (size_t index = 0; index < count; ++index) {
    double value = ctx.scoreValues.getData()[index];
    if (!std::isfinite(value))
      throw std::runtime_error("non-finite Laya score");
    maximum = std::max(maximum, value / temperature);
  }
  double sum = 0;
  for (size_t index = 0; index < count; ++index)
    sum += probabilities[index] = std::exp(
        double(ctx.scoreValues.getData()[index]) / temperature - maximum);
  for (double &value : probabilities)
    value /= sum;
  maximum = -INFINITY;
  for (size_t index = 0; index < actions; ++index) {
    if (!std::isfinite(ctx.actionValues.getData()[index]))
      throw std::runtime_error("non-finite Laya action score");
    maximum = std::max(maximum, double(ctx.actionValues.getData()[index]));
  }
  sum = 0;
  for (size_t index = 0; index < actions; ++index)
    sum += actionProbabilities[index] =
        std::exp(double(ctx.actionValues.getData()[index]) - maximum);
  for (double &value : actionProbabilities)
    value /= sum;
  const int64_t kind = ctx.qtype.getData()[0];
  const size_t answerIndex =
      std::max_element(probabilities.begin(), probabilities.end()) -
      probabilities.begin();
  double answer = answerIndex;
  if (kind == 1) {
    answer = 0;
    for (size_t index = 0; index < count; ++index)
      answer += probabilities[index] * double(index);
  } else if (kind == 2) {
    if (count != 2)
      throw std::runtime_error("Noul requires two options");
    answer = probabilities[1];
  }
  std::cout << std::setprecision(17) << "{\"type\":" << kind
            << ",\"answer_index\":" << answerIndex << ",\"answer\":" << answer
            << ",\"probabilities\":[";
  for (size_t index = 0; index < count; ++index) {
    if (index)
      std::cout << ',';
    std::cout << probabilities[index];
  }
  std::cout << "],\"action_probabilities\":[";
  for (size_t index = 0; index < actions; ++index) {
    if (index)
      std::cout << ',';
    std::cout << actionProbabilities[index];
  }
  std::cout << "]}" << std::endl;
}

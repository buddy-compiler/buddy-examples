#include "worker.h"
#include <algorithm>
#include <cstring>

void Execution::State::step(const std::vector<int64_t> &request) {
  const auto &shape = modelShape();
  auto &execution = *this;
  auto &logits = logitValues;
  const size_t start = position;
  const size_t count = request.size();
  const size_t length = start == 0 ? shape.prefill : 1;
  std::fill_n(execution.tokens, shape.prefill, int64_t(0));
  std::copy(request.begin(), request.end(), execution.tokens);
  execution.submit(0, 0, count, start, 0);
  execution.wait(0);
  std::memcpy(execution.hidden, execution.workers[0].output,
              count * shape.hidden * sizeof(float));
  for (size_t layer = 0; layer < shape.layers; ++layer) {
    for (size_t rank = 0; rank < shape.tiles; ++rank) {
      auto &worker = execution.workers[rank];
      std::fill_n(worker.input, length * shape.hidden, 0.0f);
      std::memcpy(worker.input, execution.hidden,
                  count * shape.hidden * sizeof(float));
      execution.submit(rank, 6, count, start, layer);
    }
    for (size_t rank = 0; rank < shape.tiles; ++rank)
      execution.wait(rank);
    for (size_t part = 0; part < shape.parts; ++part) {
      auto &first = execution.workers[part];
      auto &second = execution.workers[part + shape.parts];
      std::fill_n(first.input, length * shape.projection, 0.0f);
      for (size_t token = 0; token < count; ++token) {
        std::memcpy(first.input + token * shape.projection,
                    first.output + token * shape.context,
                    shape.context * sizeof(float));
        std::memcpy(first.input + token * shape.projection + shape.context,
                    second.output + token * shape.context,
                    shape.context * sizeof(float));
      }
      std::memcpy(second.input, first.input,
                  length * shape.projection * sizeof(float));
      execution.submit(part, 7, count, start, layer);
      execution.submit(part + shape.parts, 7, count, start, layer);
    }
    std::fill_n(reduced.data(), count * shape.hidden, 0.0f);
    for (size_t part = 0; part < shape.parts; ++part) {
      execution.wait(part);
      execution.wait(part + shape.parts);
      for (size_t token = 0; token < count; ++token)
        for (size_t half = 0; half < 2; ++half)
          for (size_t element = 0; element < shape.hidden / 2; ++element)
            reduced[token * shape.hidden + half * shape.hidden / 2 + element] +=
                execution.workers[part + half * shape.parts]
                    .output[token * shape.hidden / 2 + element];
    }
    for (size_t index = 0; index < count * shape.hidden; ++index)
      execution.hidden[index] += reduced[index];
    for (size_t rank = 0; rank < shape.tiles; ++rank) {
      auto &worker = execution.workers[rank];
      std::fill_n(worker.input, length * shape.hidden, 0.0f);
      std::memcpy(worker.input, execution.hidden,
                  count * shape.hidden * sizeof(float));
      execution.submit(rank, 4, count, start, layer);
    }
    for (size_t rank = 0; rank < shape.tiles; ++rank)
      execution.wait(rank);
    for (size_t part = 0; part < shape.parts; ++part) {
      auto &first = execution.workers[part];
      auto &second = execution.workers[part + shape.parts];
      std::fill_n(first.input, length * shape.intermediate, 0.0f);
      for (size_t token = 0; token < count; ++token) {
        std::memcpy(first.input + token * shape.intermediate,
                    first.output + token * shape.intermediate / 2,
                    shape.intermediate / 2 * sizeof(float));
        std::memcpy(first.input + token * shape.intermediate +
                        shape.intermediate / 2,
                    second.output + token * shape.intermediate / 2,
                    shape.intermediate / 2 * sizeof(float));
      }
      std::memcpy(second.input, first.input,
                  length * shape.intermediate * sizeof(float));
      execution.submit(part, 5, count, start, layer);
      execution.submit(part + shape.parts, 5, count, start, layer);
    }
    std::fill_n(reduced.data(), count * shape.hidden, 0.0f);
    for (size_t part = 0; part < shape.parts; ++part) {
      execution.wait(part);
      execution.wait(part + shape.parts);
      for (size_t token = 0; token < count; ++token)
        for (size_t half = 0; half < 2; ++half)
          for (size_t element = 0; element < shape.hidden / 2; ++element)
            reduced[token * shape.hidden + half * shape.hidden / 2 + element] +=
                execution.workers[part + half * shape.parts]
                    .output[token * shape.hidden / 2 + element];
    }
    for (size_t index = 0; index < count * shape.hidden; ++index)
      execution.hidden[index] += reduced[index];
  }
  const size_t localVocabulary = shape.vocabulary / shape.tiles;
  for (size_t rank = 0; rank < shape.tiles; ++rank) {
    std::memcpy(execution.workers[rank].input,
                execution.hidden + (count - 1) * shape.hidden,
                shape.hidden * sizeof(float));
    execution.submit(rank, 1, 1, 0, 0);
  }
  for (size_t rank = 0; rank < shape.tiles; ++rank) {
    execution.wait(rank);
    std::memcpy(logits.data() + rank * localVocabulary,
                execution.workers[rank].output,
                localVocabulary * sizeof(float));
  }
  position += count;
}
void Execution::prefill(const std::vector<int64_t> &tokens) {
  state->position = 0;
  state->resetCache();
  const auto &shape = modelShape();
  state->reduced.resize(shape.prefill * shape.hidden);
  state->logitValues.resize(shape.vocabulary);
  state->step(tokens);
}
void Execution::decode(int64_t token) { state->step({token}); }

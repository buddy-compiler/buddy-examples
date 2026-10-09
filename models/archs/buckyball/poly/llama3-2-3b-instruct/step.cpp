#include "execution_state.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

void Execution::State::step(const std::vector<int64_t> &request, size_t start) {
  const auto &shape = modelShape();
  const size_t count = request.size();
  const size_t length = start == 0 ? shape.prefill : 1;
  std::fill_n(tokens, shape.prefill, int64_t(0));
  std::copy(request.begin(), request.end(), tokens);
  submit(0, 0, count, start, 0);
  wait(0);
  std::memcpy(hidden, workers[0].output, count * shape.hidden * sizeof(float));
  for (size_t layer = 0; layer < shape.layers; ++layer) {
    for (uint64_t operation : {2, 3}) {
      for (size_t rank = 0; rank < shape.tiles; ++rank) {
        auto &worker = workers[rank];
        std::fill_n(worker.input, length * shape.hidden, 0.0f);
        std::memcpy(worker.input, hidden, count * shape.hidden * sizeof(float));
        submit(rank, operation, count, start, layer);
      }
      std::fill_n(reduced.data(), count * shape.hidden, 0.0f);
      for (size_t rank = 0; rank < shape.tiles; ++rank) {
        wait(rank);
        for (size_t index = 0; index < count * shape.hidden; ++index)
          reduced[index] += workers[rank].output[index];
      }
      for (size_t index = 0; index < count * shape.hidden; ++index)
        hidden[index] += reduced[index];
    }
  }
  std::memcpy(workers[0].input, hidden + (count - 1) * shape.hidden,
              shape.hidden * sizeof(float));
  submit(0, 1, 1, 0, 0);
  wait(0);
}
void Execution::prefill(const std::vector<int64_t> &tokens) {
  if (state->closed || state->prefilled || tokens.empty() ||
      tokens.size() > modelShape().prefill)
    throw std::runtime_error("invalid Llama prefill request");
  state->step(tokens, 0);
  state->position = tokens.size();
  state->prefilled = true;
}
void Execution::decode(int64_t token) {
  if (state->closed || !state->prefilled ||
      state->position >= modelShape().cache)
    throw std::runtime_error("invalid Llama decode request");
  state->step(std::vector<int64_t>{token}, state->position);
  ++state->position;
}
std::span<const float> Execution::logits() const {
  if (state->closed || !state->prefilled)
    throw std::runtime_error("Llama logits are unavailable");
  return {state->workers[0].output, modelShape().vocabulary};
}
const Request &Execution::request() const { return state->request; }
void Execution::close() { state->close(); }
Execution::~Execution() noexcept(false) {
  if (!state->closed)
    state->close();
}

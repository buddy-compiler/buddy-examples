#include "state.h"
#include <runtime.h>

extern "C" void _mlir_ciface_forward_prefill_embedding(Hidden *, Floats *, Tokens *, Slots *);
extern "C" void _mlir_ciface_forward_decode_embedding(Hidden *, Floats *, Tokens *, Slots *);
extern "C" void _mlir_ciface_forward_prefill_norm(Matrix *, Floats *, Matrix *);
extern "C" void _mlir_ciface_forward_decode_norm(Matrix *, Floats *, Matrix *);
extern "C" void _mlir_ciface_forward_output(Matrix *, Bytes *, Matrix *);

void Thinker::embedding(size_t count) {
  size_t length = count == 1 ? 1 : prefillLength;
  Tokens tokens({1, length}, int64_t(0));
  read_values(tokens.getData(), count);
  for (size_t i = 0; i < count; ++i)
    if (tokens[i] < 0 || size_t(tokens[i]) >= vocabulary) throw std::runtime_error("invalid token ID");
  int64_t first_token = rank * vocabularyPart;
  View<int64_t, 1> begin(&first_token, {1});
  auto fp = float_parameters(0);
  Hidden result({1, length, hiddenSize}, false, 0);
  auto run = count == 1 ? _mlir_ciface_forward_decode_embedding : _mlir_ciface_forward_prefill_embedding;
  run(&result, &fp, &tokens, &begin);
  write_values(result.getData(), count * hiddenSize);
  workspace_free(result.release());
}

void Thinker::reduce(size_t count, bool residual) {
  std::vector<float> result(count * hiddenSize, 0.0f), partial(result.size());
  if (residual) read_values(result.data(), result.size());
  for (size_t rank = 0; rank < parts; ++rank) {
    read_values(partial.data(), partial.size());
    for (size_t i = 0; i < result.size(); ++i) result[i] += partial[i];
  }
  write_values(result.data(), result.size());
  exchange = std::move(result);
}

void Thinker::norm(size_t count) {
  size_t length = count == 1 ? 1 : prefillLength;
  Matrix hidden({length, hiddenSize}, 0.0f), result({length, hiddenSize}, false, 0);
  read_values(hidden.getData(), count * hiddenSize);
  auto fp = float_parameters(2);
  auto run = count == 1 ? _mlir_ciface_forward_decode_norm : _mlir_ciface_forward_prefill_norm;
  run(&result, &fp, &hidden);
  write_values(result.getData(), count * hiddenSize);
  workspace_free(result.release());
}

void Thinker::output() {
  Matrix hidden({1, hiddenSize}), result({1, vocabularyPart}, false, 0);
  read_values(hidden.getData(), hiddenSize);
  auto packed = byte_parameters(1);
  _mlir_ciface_forward_output(&result, &packed, &hidden);
  write_values(result.getData(), vocabularyPart);
  workspace_free(result.release());
}

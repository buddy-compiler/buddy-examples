#include "state.h"
#include <runtime.h>

extern "C" void _mlir_ciface_forward_prefill_router(RouterResult *, Floats *, Matrix *);
extern "C" void _mlir_ciface_forward_decode_router(RouterResult *, Floats *, Matrix *);
extern "C" void _mlir_ciface_forward_expert(Matrix *, Bytes *, Matrix *);

void Thinker::router(size_t count, size_t layer) {
  size_t length = count == 1 ? 1 : prefillLength;
  Matrix hidden({length, hiddenSize}, 0.0f);
  read_values(hidden.getData(), count * hiddenSize);
  auto fp = float_parameters(layer_region(layer) + 1);
  RouterResult result{Matrix({length, hiddenSize}, false, 0),
                      Tokens({length, topK}, false, 0), Matrix({length, topK}, false, 0)};
  auto run = count == 1 ? _mlir_ciface_forward_decode_router : _mlir_ciface_forward_prefill_router;
  run(&result, &fp, &hidden);
  write_values(result.hidden.getData(), count * hiddenSize);
  write_values(result.experts.getData(), count * topK);
  write_values(result.scores.getData(), count * topK);
  workspace_free(result.hidden.release());
  workspace_free(result.experts.release());
  workspace_free(result.scores.release());
}

void Thinker::experts(size_t count, size_t layer) {
  size_t first_expert = layer_region(layer) + 2;
  Matrix hidden({count, hiddenSize}), scores({count, topK}), output({count, hiddenSize}, 0.0f);
  Tokens selected({count, topK});
  read_values(hidden.getData(), count * hiddenSize);
  read_values(selected.getData(), count * topK);
  read_values(scores.getData(), count * topK);
  for (size_t token = 0; token < count; ++token) {
    View<float, 2> input(hidden.getData() + token * hiddenSize, {1, hiddenSize});
    for (size_t choice = 0; choice < topK; ++choice) {
      auto expert = selected[token * topK + choice];
      if (expert < 0 || size_t(expert) >= expertsCount) throw std::runtime_error("invalid expert index");
      auto weights = byte_parameters(first_expert + expert);
      Matrix partial({1, hiddenSize}, false, 0);
      _mlir_ciface_forward_expert(&partial, &weights, &input);
      for (size_t i = 0; i < hiddenSize; ++i)
        output[token * hiddenSize + i] += partial[i] * scores[token * topK + choice];
      workspace_free(partial.release());
    }
  }
  write_values(output.getData(), count * hiddenSize);
}

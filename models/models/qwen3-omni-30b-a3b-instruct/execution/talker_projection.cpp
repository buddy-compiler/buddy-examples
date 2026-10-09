#include "embedding.h"
#include "talker.h"
#include <runtime.h>

using namespace TalkerParams;
using Projection = void (*)(Matrix *, Floats *, Bytes *, Matrix *);
extern "C" void _mlir_ciface_forward_talker_prefill_text_projection(Matrix *,
                                                                    Floats *,
                                                                    Bytes *,
                                                                    Matrix *);
extern "C" void _mlir_ciface_forward_talker_decode_text_projection(Matrix *,
                                                                   Floats *,
                                                                   Bytes *,
                                                                   Matrix *);
extern "C" void _mlir_ciface_forward_talker_prefill_hidden_projection(Matrix *,
                                                                      Floats *,
                                                                      Bytes *,
                                                                      Matrix *);
extern "C" void _mlir_ciface_forward_talker_decode_hidden_projection(Matrix *,
                                                                     Floats *,
                                                                     Bytes *,
                                                                     Matrix *);
extern "C" void _mlir_ciface_forward_talker_output(Matrix *, Bytes *, Matrix *);
extern "C" void _mlir_ciface_forward_predictor_output(Matrix *, Bytes *,
                                                      Matrix *);

void Talker::embedding(size_t count, size_t group) {
  if (group >= groups)
    throw std::runtime_error("invalid codec embedding group");
  const auto &region = regions.at(group);
  const size_t stride = width + width / 32;
  if (region.float_count || region.byte_count % stride)
    throw std::runtime_error("codec embedding requires MXFP8 row storage");
  const size_t rows = region.byte_count / stride;
  std::vector<uint64_t> indices(count);
  read_values(indices.data(), count);
  for (uint64_t index : indices)
    if (index >= rows)
      throw std::runtime_error("invalid codec token ID");
  std::vector<float> decoded(count * width);
  embedding_rows(decoded.data(), bytes.get() + region.byte_offset, width,
                 indices.data(), count);
  write_values(decoded.data(), decoded.size());
}

void Talker::resize(size_t count, size_t kind) {
  if (kind > 1)
    throw std::runtime_error("invalid Talker projection kind");
  size_t length = count == 1 ? 1 : prefill;
  Matrix hidden({length, thinkerWidth}, 0.0f),
      result({length, width}, false, 0);
  read_values(hidden.getData(), count * thinkerWidth);
  auto [fp, packed] = parameters(groups + kind);
  Projection run =
      kind == 0
          ? (count == 1 ? _mlir_ciface_forward_talker_decode_text_projection
                        : _mlir_ciface_forward_talker_prefill_text_projection)
          : (count == 1
                 ? _mlir_ciface_forward_talker_decode_hidden_projection
                 : _mlir_ciface_forward_talker_prefill_hidden_projection);
  run(&result, &fp, &packed, &hidden);
  write_values(result.getData(), count * width);
  workspace_free(result.release());
}

void Talker::logits(size_t group, bool predictor) {
  Matrix hidden({1, width}),
      result({1, predictor ? predictorVocabulary : vocabulary}, false, 0);
  read_values(hidden.getData(), width);
  auto [fp, packed] = parameters(predictor ? groups + 5 + group : groups + 3);
  auto run = predictor ? _mlir_ciface_forward_predictor_output
                       : _mlir_ciface_forward_talker_output;
  run(&result, &packed, &hidden);
  write_values(result.getData(), predictor ? predictorVocabulary : vocabulary);
  workspace_free(result.release());
}

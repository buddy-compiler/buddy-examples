#include "talker.h"
#include <algorithm>
#include <runtime.h>

using namespace TalkerParams;
using AttentionKernel = void (*)(AttentionResult *, Floats *, Bytes *, Hidden *,
                                 Cache *, Cache *, Slots *, Positions *);
using DenseKernel = void (*)(Matrix *, Floats *, Bytes *, Matrix *);
using NormKernel = void (*)(Matrix *, Floats *, Matrix *);
extern "C" void
_mlir_ciface_forward_predictor_prefill_attention(AttentionResult *, Floats *,
                                                 Bytes *, Hidden *, Cache *,
                                                 Cache *, Slots *, Positions *);
extern "C" void
_mlir_ciface_forward_predictor_decode_attention(AttentionResult *, Floats *,
                                                Bytes *, Hidden *, Cache *,
                                                Cache *, Slots *, Positions *);
extern "C" void _mlir_ciface_forward_predictor_prefill_dense(Matrix *, Floats *,
                                                             Bytes *, Matrix *);
extern "C" void _mlir_ciface_forward_predictor_decode_dense(Matrix *, Floats *,
                                                            Bytes *, Matrix *);
extern "C" void _mlir_ciface_forward_talker_prefill_norm(Matrix *, Floats *,
                                                         Matrix *);
extern "C" void _mlir_ciface_forward_talker_decode_norm(Matrix *, Floats *,
                                                        Matrix *);
extern "C" void _mlir_ciface_forward_predictor_prefill_norm(Matrix *, Floats *,
                                                            Matrix *);
extern "C" void _mlir_ciface_forward_predictor_decode_norm(Matrix *, Floats *,
                                                           Matrix *);

void Talker::forward(size_t count, size_t start, bool predictor) {
  size_t length = count == 1 ? 1 : (predictor ? 2 : prefill);
  size_t limit = predictor ? predictorCapacity : capacity;
  size_t &next = predictor ? predictor_position : position;
  if (start + length > limit || (start && start != next))
    throw std::runtime_error("non-contiguous or overflowing Talker cache");
  std::vector<float> hidden(length * width, 0.0f);
  read_values(hidden.data(), count * width);
  Slots slots({length});
  Positions positions({3, length}, int64_t(0));
  for (size_t index = 0; index < length; ++index)
    slots[index] = start + index;
  for (size_t axis = 0; axis < 3; ++axis)
    read_values(positions.getData() + axis * length, count);
  auto &k = predictor ? predictor_keys : keys;
  auto &v = predictor ? predictor_values : values;
  size_t base = 2 * groups + 4;
  size_t first = predictor ? base + layerCount * (expertCount + 3) : base;
  size_t stride = predictor ? 2 : expertCount + 3;
  size_t attentionLimit = limit, bucket = 0;
  if (!predictor) {
    bucket = std::lower_bound(std::begin(cacheBuckets), std::end(cacheBuckets),
                              start + length) -
             std::begin(cacheBuckets);
    attentionLimit = cacheBuckets[bucket];
  }
  AttentionKernel attend =
      predictor
          ? (count == 1 ? _mlir_ciface_forward_predictor_decode_attention
                        : _mlir_ciface_forward_predictor_prefill_attention)
          : (count == 1 ? decodeAttention[bucket] : prefillAttention[bucket]);
  for (size_t layer = 0; layer < (predictor ? predictorLayers : layerCount);
       ++layer) {
    workspace_begin(workspace, 64 * 1024 * 1024);
    size_t index = first + layer * stride;
    auto [fp, packed] = parameters(index);
    View<float, 3> input(hidden.data(), {1, length, width});
    size_t heads = predictor ? predictorKVHeads : kvHeads;
    std::unique_ptr<Cache> croppedKeys, croppedValues;
    Cache *inputKeys = &k[layer], *inputValues = &v[layer];
    if (attentionLimit != limit) {
      croppedKeys = std::make_unique<Cache>(
          std::vector<size_t>{1, heads, attentionLimit, headDim});
      croppedValues = std::make_unique<Cache>(
          std::vector<size_t>{1, heads, attentionLimit, headDim});
      for (size_t head = 0; head < heads; ++head) {
        std::copy_n(k[layer].getData() + head * limit * headDim,
                    attentionLimit * headDim,
                    croppedKeys->getData() + head * attentionLimit * headDim);
        std::copy_n(v[layer].getData() + head * limit * headDim,
                    attentionLimit * headDim,
                    croppedValues->getData() + head * attentionLimit * headDim);
      }
      inputKeys = croppedKeys.get();
      inputValues = croppedValues.get();
    }
    AttentionResult result{Hidden({1, length, width}, false, 0),
                           Cache({1, predictor ? predictorKVHeads : kvHeads,
                                  attentionLimit, headDim},
                                 false, 0),
                           Cache({1, predictor ? predictorKVHeads : kvHeads,
                                  attentionLimit, headDim},
                                 false, 0)};
    attend(&result, &fp, &packed, &input, inputKeys, inputValues, &slots,
           &positions);
    for (size_t head = 0; head < (predictor ? predictorKVHeads : kvHeads);
         ++head) {
      size_t offset = (head * limit + start) * headDim;
      size_t sourceOffset = (head * attentionLimit + start) * headDim;
      std::copy_n(result.keys.getData() + sourceOffset, length * headDim,
                  k[layer].getData() + offset);
      std::copy_n(result.values.getData() + sourceOffset, length * headDim,
                  v[layer].getData() + offset);
    }
    for (size_t i = 0; i < count * width; ++i)
      hidden[i] += result.hidden[i];
    workspace_free(result.hidden.release());
    workspace_free(result.keys.release());
    workspace_free(result.values.release());
    croppedKeys.reset();
    croppedValues.reset();
    workspace_begin(workspace, 64 * 1024 * 1024);
    View<float, 2> matrix(hidden.data(), {length, width});
    if (predictor) {
      auto [df, dp] = parameters(index + 1);
      Matrix output({length, width}, false, 0);
      DenseKernel run = count == 1
                            ? _mlir_ciface_forward_predictor_decode_dense
                            : _mlir_ciface_forward_predictor_prefill_dense;
      run(&output, &df, &dp, &matrix);
      for (size_t i = 0; i < count * width; ++i)
        hidden[i] += output[i];
      workspace_free(output.release());
    } else {
      experts(matrix, count, layer);
    }
  }
  workspace_begin(workspace, 64 * 1024 * 1024);
  auto [fp, packed] = parameters(predictor ? groups + 4 : groups + 2);
  View<float, 2> matrix(hidden.data(), {length, width});
  Matrix normalized({length, width}, false, 0);
  NormKernel normalize =
      predictor ? (count == 1 ? _mlir_ciface_forward_predictor_decode_norm
                              : _mlir_ciface_forward_predictor_prefill_norm)
                : (count == 1 ? _mlir_ciface_forward_talker_decode_norm
                              : _mlir_ciface_forward_talker_prefill_norm);
  normalize(&normalized, &fp, &matrix);
  write_values(normalized.getData(), count * width);
  workspace_free(normalized.release());
  next = start + count;
}

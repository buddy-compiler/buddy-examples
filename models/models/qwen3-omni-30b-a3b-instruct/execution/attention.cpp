#include "state.h"
#include <runtime.h>

extern "C" void _mlir_ciface_forward_prefill_attention(AttentionResult *,
                                                       Floats *, Bytes *,
                                                       Hidden *, Cache *,
                                                       Cache *, Slots *,
                                                       Positions *);
extern "C" void _mlir_ciface_forward_decode_attention(AttentionResult *,
                                                      Floats *, Bytes *,
                                                      Hidden *, Cache *,
                                                      Cache *, Slots *,
                                                      Positions *);

void Thinker::attention(size_t count, size_t start, size_t layer) {
  size_t length = count == 1 ? 1 : prefillLength;
  size_t parameters = layer_region(layer), local = layer - first;
  if (start + length > cacheLength ||
      (start != 0 && start != cache_tokens[local]))
    throw std::runtime_error(
        "attention cache position is not contiguous or exceeds capacity");
  Hidden hidden({1, length, hiddenSize}, 0.0f);
  Slots slots({length});
  Positions positions({3, length}, int64_t(0));
  read_values(hidden.getData(), count * hiddenSize);
  for (size_t axis = 0; axis < 3; ++axis)
    read_values(positions.getData() + axis * length, count);
  for (size_t i = 0; i < length; ++i)
    slots[i] = start + i;
  auto fp = float_parameters(parameters);
  auto packed = byte_parameters(parameters);
  const std::vector<size_t> shape{1, kvHeads, cacheLength, headSize};
  AttentionResult result{Hidden({1, length, hiddenSize}, false, 0),
                         Cache(shape, false, 0), Cache(shape, false, 0)};
  auto run = count == 1 ? _mlir_ciface_forward_decode_attention
                        : _mlir_ciface_forward_prefill_attention;
  run(&result, &fp, &packed, &hidden, &keys[local], &values[local], &slots,
      &positions);
  for (size_t head = 0; head < kvHeads; ++head) {
    size_t offset = (head * cacheLength + start) * headSize;
    std::copy_n(result.keys.getData() + offset, length * headSize,
                keys[local].getData() + offset);
    std::copy_n(result.values.getData() + offset, length * headSize,
                values[local].getData() + offset);
  }
  cache_tokens[local] = start + count;
  write_values(result.hidden.getData(), count * hiddenSize);
  workspace_free(result.hidden.release());
  workspace_free(result.keys.release());
  workspace_free(result.values.release());
}

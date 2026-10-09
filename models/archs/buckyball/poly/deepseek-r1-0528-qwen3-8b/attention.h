#pragma once
#include "weights.h"
#include <buddy/Core/Container.h>
#include <cstdint>

using Hidden = MemRef<float, 3>;
using Cache = MemRef<int8_t, 4>;
using Floats = MemRef<float, 1>;
using Bytes = MemRef<int8_t, 1>;
using Positions = MemRef<int64_t, 1>;

struct AttentionFloatView {
  float *allocated, *aligned;
  intptr_t offset, size, stride;
};
struct AttentionWeightView {
  int8_t *allocated, *aligned;
  intptr_t offset, size, stride;
};
struct AttentionBodyResult {
  Cache keyCodes, keyScales, valueCodes, valueScales;
  Hidden context;
};
using AttentionBodyKernel =
    void (*)(AttentionBodyResult *, Hidden *, AttentionFloatView *,
             AttentionWeightView *, AttentionFloatView *, AttentionWeightView *,
             AttentionFloatView *, AttentionWeightView *, Positions *,
             AttentionFloatView *, Cache *, Cache *, Cache *, Cache *);
using AttentionProjectionKernel = void (*)(Hidden *, Hidden *,
                                           AttentionWeightView *);
struct AttentionKernels {
  size_t headSize;
  WeightRange query, key, value, projectionWeights;
  AttentionBodyKernel body;
  AttentionProjectionKernel projection;
};

void runAttentionBody(const AttentionKernels &kernels,
                      AttentionBodyResult *result, Floats *floats,
                      Bytes *weights, Hidden *hidden, Cache *keyCodes,
                      Cache *keyScales, Cache *valueCodes, Cache *valueScales,
                      Positions *positions);
void runAttentionProjection(const AttentionKernels &kernels, Hidden *result,
                            Bytes *weights, Hidden *context);

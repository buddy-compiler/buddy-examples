#pragma once
#include "weights.h"
#include <array>
#include <buddy/Core/Container.h>
#include <cstdint>

using Hidden = MemRef<float, 3>;
using Floats = MemRef<float, 1>;
using Bytes = MemRef<int8_t, 1>;

struct FfnWeightView {
  int8_t *allocated, *aligned;
  intptr_t offset, size, stride;
};
using FfnExpandKernel = void (*)(Hidden *, Hidden *, Floats *, FfnWeightView *,
                                 FfnWeightView *);
using FfnDownKernel = void (*)(Hidden *, Hidden *, FfnWeightView *);
struct FfnKernels {
  size_t hidden, intermediate;
  std::array<WeightRange, 3> gateWeights, upWeights, downWeights;
  std::array<size_t, 3> expandWidths, downWidths;
  std::array<FfnExpandKernel, 3> expand;
  std::array<FfnDownKernel, 3> down;
};

void runFfnExpand(const FfnKernels &kernels, Hidden *result, Floats *norm,
                  Bytes *weights, Hidden *input);
void runFfnDown(const FfnKernels &kernels, Hidden *result, Bytes *weights,
                Hidden *input);

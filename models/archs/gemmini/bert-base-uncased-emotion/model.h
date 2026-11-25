#pragma once

#include "MemRef.h"

namespace bert {
using Parameters = model::MemRef<const float, 1>;
using Bias = model::MemRef<const int32_t, 1>;
using Weights = model::MemRef<const int8_t, 1>;
using Tokens = model::MemRef<const int64_t, 2>;
using Output = model::MemRef<float, 2>;
} // namespace bert

extern "C" void _mlir_ciface_forward(bert::Output *, bert::Parameters *,
                                     bert::Bias *, bert::Weights *,
                                     bert::Tokens *, bert::Tokens *,
                                     bert::Tokens *, bert::Tokens *);

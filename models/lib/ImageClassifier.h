#pragma once

#include "MemRef.h"

namespace model {
using Parameters = MemRef<const float, 1>;
using Bias = MemRef<const int32_t, 1>;
using Weights = MemRef<const int8_t, 1>;
using Image = MemRef<const float, 4>;
using Scores = MemRef<float, 2>;
} // namespace model

extern "C" void _mlir_ciface_forward(model::Scores *, model::Parameters *,
                                     model::Bias *, model::Weights *,
                                     model::Image *);
namespace model {
using Forward = decltype(&_mlir_ciface_forward);
}

#pragma once

#include "MemRef.h"

namespace yolo {
using Parameters = model::MemRef<const float, 1>;
using Bias = model::MemRef<const int32_t, 1>;
using Weights = model::MemRef<const int8_t, 1>;
using Input = model::MemRef<const float, 4>;
using Output = model::MemRef<float, 3>;
} // namespace yolo

extern "C" void _mlir_ciface_forward(yolo::Output *, yolo::Parameters *,
                                     yolo::Bias *, yolo::Weights *,
                                     yolo::Input *);
namespace yolo {
using Forward = decltype(&_mlir_ciface_forward);
}

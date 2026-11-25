#pragma once

#include "MemRef.h"

namespace mobilenet {

// The compiled MLIR C interface. Storage ownership stays with the caller;
// forward returns an owned output allocation, released with free().
using model::MemRef;

using Parameters = MemRef<const float, 1>;
using Bias = MemRef<const int32_t, 1>;
using Weights = MemRef<const int8_t, 1>;
using Input = MemRef<const float, 4>;
using Output = MemRef<float, 2>;

} // namespace mobilenet

extern "C" void _mlir_ciface_forward(mobilenet::Output *,
                                     mobilenet::Parameters *, mobilenet::Bias *,
                                     mobilenet::Weights *, mobilenet::Input *);
namespace mobilenet {
using Forward = decltype(&_mlir_ciface_forward);
}

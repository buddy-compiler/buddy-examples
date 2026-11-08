#pragma once

#include "MemRef.h"

namespace mixer {
using Parameters = model::MemRef<const float, 1>;
using Bias = model::MemRef<const int32_t, 1>;
using Weights = model::MemRef<const int8_t, 1>;
using Input = model::MemRef<const float, 2>;
using Output = model::MemRef<float, 2>;

struct Arguments {
  Output *output;
  Parameters *parameters;
  Bias *bias;
  Weights *weights;
  Input *input;
};
} // namespace mixer

extern "C" void _mlir_ciface_forward(mixer::Output *, mixer::Parameters *,
                                     mixer::Bias *, mixer::Weights *,
                                     mixer::Input *);
namespace mixer {
using Forward = decltype(&_mlir_ciface_forward);
}

#pragma once
#include "MemRef.h"

namespace laya {
struct Input {
  int64_t tokens[128], mask[128], positions[16];
  bool valid[16];
  int64_t type[1];
};
static_assert(sizeof(Input) == 2200);
using Parameters = model::MemRef<const float, 1>;
using Bias = model::MemRef<const int32_t, 1>;
using Weights = model::MemRef<const int8_t, 1>;
using Tokens = model::MemRef<const int64_t, 2>;
using Valid = model::MemRef<const bool, 2>;
using Type = model::MemRef<const int64_t, 1>;
struct Output {
  model::MemRef<float, 2> choice, action;
};
struct Arguments {
  Output *output;
  Parameters *parameters;
  Bias *bias;
  Weights *weights;
  Tokens *tokens, *mask, *positions;
  Valid *valid;
  Type *type;
};
} // namespace laya
extern "C" void _mlir_ciface_forward(laya::Output *, laya::Parameters *,
                                     laya::Bias *, laya::Weights *,
                                     laya::Tokens *, laya::Tokens *,
                                     laya::Tokens *, laya::Valid *,
                                     laya::Type *);
namespace laya {
using Forward = decltype(&_mlir_ciface_forward);
}

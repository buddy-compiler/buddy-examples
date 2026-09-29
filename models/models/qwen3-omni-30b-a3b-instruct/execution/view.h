#pragma once
#include <buddy/Core/Container.h>
#include <array>

template <typename T, size_t Rank>
struct View : MemRef<T, Rank> {
  View(T *data, std::array<size_t, Rank> shape) {
    this->aligned = data;
    std::copy(shape.begin(), shape.end(), this->sizes);
    this->setStrides();
  }
};

using Floats = MemRef<float, 1>;
using Bytes = MemRef<int8_t, 1>;
using Matrix = MemRef<float, 2>;
using Hidden = MemRef<float, 3>;
using Cache = MemRef<float, 4>;
using Tokens = MemRef<int64_t, 2>;
using Slots = MemRef<int64_t, 1>;
using Positions = MemRef<int64_t, 2>;
struct AttentionResult { Hidden hidden; Cache keys, values; };
struct RouterResult { Matrix hidden; Tokens experts; Matrix scores; };

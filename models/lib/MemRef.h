#pragma once

#include <stdint.h>

namespace model {

// MLIR C ABI descriptor. Ownership is managed by the caller.
template <typename T, unsigned Rank> struct MemRef {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Rank];
  int64_t strides[Rank];

  T *data() const { return aligned + offset; }
};

} // namespace model

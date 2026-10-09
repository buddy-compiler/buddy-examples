#include "embedding.h"
#include <CRunnerUtils.h>
#include <cstring>
#include <stdexcept>
#include <vector>
extern "C" void _mlir_ciface_rvv_mxfp8_decode(UnrankedMemRefType<float> *,
                                              UnrankedMemRefType<int8_t> *,
                                              UnrankedMemRefType<int8_t> *);
void embedding_rows(float *output, const int8_t *table, size_t width,
                    const uint64_t *indices, size_t count) {
  if (!count || !width || width % 32)
    throw std::runtime_error("invalid MXFP8 embedding shape");
  const int64_t columns = width, rows = count, stride = width + width / 32;
  std::vector<int8_t> selected(count * stride);
  for (size_t row = 0; row < count; ++row)
    std::memcpy(selected.data() + row * stride, table + indices[row] * stride,
                stride);
  StridedMemRefType<int8_t, 2> codes{
      selected.data(), selected.data(), 0, {rows, columns}, {stride, 1}};
  StridedMemRefType<int8_t, 2> scales{selected.data(),
                                      selected.data(),
                                      columns,
                                      {rows, columns / 32},
                                      {stride, 1}};
  StridedMemRefType<float, 2> result{
      output, output, 0, {rows, columns}, {columns, 1}};
  UnrankedMemRefType<int8_t> c{2, &codes}, s{2, &scales};
  UnrankedMemRefType<float> out{2, &result};
  _mlir_ciface_rvv_mxfp8_decode(&out, &c, &s);
}

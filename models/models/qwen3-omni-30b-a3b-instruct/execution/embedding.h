#pragma once
#include <cstddef>
#include <cstdint>
void embedding_rows(float *output, const int8_t *table, size_t width,
                    const uint64_t *indices, size_t count);

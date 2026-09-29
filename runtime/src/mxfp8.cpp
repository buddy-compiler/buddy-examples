#include <CRunnerUtils.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static uint32_t roundedShift(uint32_t value, unsigned shift) {
  if (shift > 31)
    return 0;
  uint32_t high = value >> shift;
  uint32_t low = value & ((uint32_t(1) << shift) - 1);
  uint32_t half = uint32_t(1) << (shift - 1);
  return high + (low > half || (low == half && (high & 1)));
}

static uint8_t encode(uint32_t bits, int blockExponent) {
  uint8_t sign = (bits >> 24) & 128;
  if (!(bits & 0x7fffffff))
    return sign;
  int exponent = int((bits >> 23) & 255);
  uint32_t fraction = bits & 0x7fffff;
  if (exponent == 0) {
    int shift = __builtin_clz(fraction) - 8;
    fraction = (fraction << shift) & 0x7fffff;
    exponent = 1 - shift;
  }
  exponent -= blockExponent + 120;
  uint32_t code =
      exponent <= 0 ? roundedShift(fraction | 0x800000, unsigned(21 - exponent))
                    : unsigned(exponent * 8) + roundedShift(fraction, 20);
  return sign | std::min(code, 126u);
}

extern "C" void _mlir_ciface_mxfp8_quant(StridedMemRefType<float, 2> *input,
                                         StridedMemRefType<int8_t, 1> *output,
                                         int64_t tileRows, int64_t tileK) {
  int64_t m = input->sizes[0], k = input->sizes[1];
  int64_t mt = (m + tileRows - 1) / tileRows, kt = (k + tileK - 1) / tileK;
  int64_t bytes = tileRows * tileK + tileRows * tileK / 32;
  if (k % 32 || tileK % 32 || output->sizes[0] != mt * kt * bytes) {
    fprintf(stderr, "mxfp8: invalid activation packing shape\n");
    abort();
  }
  auto *source = input->data + input->offset;
  auto *dest = output->data + output->offset;
  const int64_t rowStride = input->strides[0], columnStride = input->strides[1];
  const int64_t outputStride = output->strides[0];
  for (int64_t mr = 0; mr < mt; ++mr)
    for (int64_t kr = 0; kr < kt; ++kr)
      for (int64_t row = 0; row < tileRows; ++row)
        for (int64_t block = 0; block < tileK / 32; ++block) {
          int64_t r = mr * tileRows + row, c = kr * tileK + block * 32;
          int64_t base = (mr * kt + kr) * bytes;
          int64_t codes = base + row * tileK + block * 32;
          int64_t scale = base + tileRows * tileK + row * (tileK / 32) + block;
          if (r >= m || c >= k) {
            dest[scale * outputStride] = 127;
            for (int i = 0; i < 32; ++i) dest[(codes + i) * outputStride] = 0;
            continue;
          }
          uint32_t values[32], maximum = 0;
          const float *blockInput = source + r * rowStride + c * columnStride;
          for (int i = 0; i < 32; ++i) {
            memcpy(&values[i], blockInput + i * columnStride, 4);
            uint32_t magnitude = values[i] & 0x7fffffff;
            if (magnitude >= 0x7f800000) {
              fprintf(stderr, "mxfp8: non-finite activation\n");
              abort();
            }
            maximum = std::max(maximum, magnitude);
          }
          int exponent =
              maximum == 0 ? 0 : std::max(-127, int(maximum >> 23) - 135);
          dest[scale * outputStride] = uint8_t(exponent + 127);
          for (int i = 0; i < 32; ++i)
            dest[(codes + i) * outputStride] = encode(values[i], exponent);
        }
}

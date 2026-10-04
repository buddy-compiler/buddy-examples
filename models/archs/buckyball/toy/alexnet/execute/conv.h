#ifndef ALEXNET_GEMMINI_CONV_H
#define ALEXNET_GEMMINI_CONV_H
#include <stdint.h>
typedef struct { float *allocated, *data; int64_t offset, sizes[4], strides[4]; } AlexnetF32_4;
typedef struct { int8_t *allocated, *data; int64_t offset, sizes[2], strides[2]; } AlexnetI8_2;
typedef struct { float *allocated, *data; int64_t offset, sizes[1], strides[1]; } AlexnetF32_1;
void _mlir_ciface_gemmini_conv(AlexnetF32_4 *, AlexnetI8_2 *, AlexnetF32_1 *,
                               AlexnetF32_4 *, int64_t, int64_t, int64_t, int64_t,
                               float, float);
#endif

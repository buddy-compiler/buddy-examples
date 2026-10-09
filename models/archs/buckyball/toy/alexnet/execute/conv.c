#include "conv.h"
#include <bbhw/isa/isa.h>
#include <isa/gemmini.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void touch_pages(int8_t *data, size_t bytes) {
  // DMA cannot demand-fault Linux pages, including zero-only calloc padding.
  volatile int8_t *memory = data;
  for (size_t index = 0; index < bytes; index += 4096)
    memory[index] = memory[index];
  memory[bytes - 1] = memory[bytes - 1];
}

void _mlir_ciface_gemmini_conv(AlexnetF32_4 *input, AlexnetI8_2 *weight,
                               AlexnetF32_1 *bias, AlexnetF32_4 *output,
                               int64_t kh, int64_t kw, int64_t stride,
                               int64_t padding, float inputScale, float weightScale) {
  int64_t batches = input->sizes[0], channels = input->sizes[1];
  int64_t height = input->sizes[2], width = input->sizes[3];
  int64_t oh = output->sizes[2], ow = output->sizes[3];
  int64_t n = weight->sizes[1], k = channels * kh * kw, m = batches * oh * ow;
  if (k != weight->sizes[0] || n != bias->sizes[0] ||
      output->sizes[0] != batches || output->sizes[1] != n ||
      oh != (height + 2 * padding - kh) / stride + 1 ||
      ow != (width + 2 * padding - kw) / stride + 1 ||
      k > INT32_MAX / (128 * 128) || weight->strides[1] != 1 ||
      !isfinite(inputScale) || inputScale <= 0 ||
      !isfinite(weightScale) || weightScale <= 0) abort();
  int64_t kTiles = (k + 15) / 16;
  size_t panelBytes = (m + 15) / 16 * kTiles * 256;
  int8_t *panels = calloc(panelBytes, 1);
  if (!panels) abort();
  fprintf(stderr, "[AlexNet] Gemmini M=%ld N=%ld K=%ld\n", (long)m, (long)n, (long)k);
  // Expand and quantize each input value once, then reuse it for every N tile.
  for (int64_t row = 0; row < m; ++row) {
    int64_t batch = row / (oh * ow), y0 = row / ow % oh * stride - padding;
    int64_t x0 = row % ow * stride - padding;
    for (int64_t inner = 0; inner < k; ++inner) {
      int64_t channel = inner % channels, y = y0 + inner / (kw * channels);
      int64_t x = x0 + inner / channels % kw;
      if (y < 0 || y >= height || x < 0 || x >= width) continue;
      float value = input->data[input->offset + batch * input->strides[0] +
                               channel * input->strides[1] + y * input->strides[2] +
                               x * input->strides[3]];
      if (!isfinite(value)) abort();
      float scaled = value / inputScale;
      scaled = scaled < -128.f ? -128.f : scaled > 127.f ? 127.f : scaled;
      int32_t code;
      __asm__ volatile("fcvt.w.s %0, %1, rne" : "=r"(code) : "f"(scaled));
      panels[(row / 16 * kTiles + inner / 16) * 256 + row % 16 * 16 + inner % 16] = code;
    }
  }
  touch_pages(panels, panelBytes);
  int8_t tail[256] __attribute__((aligned(64)));
  int32_t result[256] __attribute__((aligned(64))) = {0};
  bb_mem_alloc(0, 1, 1); bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1); bb_mem_alloc(3, 1, 4);
  bb_mset_clear(2, 1, 1);
  bb_gemmini_config(1, 0, 0, 0, 0);
  for (int64_t row = 0; row < m; row += 16)
    for (int64_t col = 0; col < n; col += 16) {
      for (int64_t inner = 0; inner < k; inner += 16) {
        bb_mvin((uintptr_t)(panels + (row / 16 * kTiles + inner / 16) * 256), 1, 16, 1);
        if (inner + 16 <= k && col + 16 <= n && weight->strides[0] % 16 == 0) {
          bb_mvin((uintptr_t)(weight->data + weight->offset + inner * weight->strides[0] + col),
                  0, 16, weight->strides[0] / 16);
        } else {
          for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j)
              tail[i * 16 + j] = inner + i < k && col + j < n
                  ? weight->data[weight->offset + (inner + i) * weight->strides[0] + col + j] : 0;
          bb_mvin((uintptr_t)tail, 0, 16, 1);
        }
        if (inner == 0) {
          bb_gemmini_preload(0, 3, 16, 0, 0);
          bb_gemmini_compute_preloaded(1, 2, 3, 16, 0, 0, 0);
        } else bb_gemmini_compute_accumulated(1, 0, 3, 16, 0, 0, 0);
      }
      bb_mvout((uintptr_t)result, 3, 16, 1);
      bb_fence();
      for (int i = 0; i < 16 && row + i < m; ++i)
        for (int j = 0; j < 16 && col + j < n; ++j) {
          int64_t index = row + i, batch = index / (oh * ow), y = index / ow % oh, x = index % ow;
          output->data[output->offset + batch * output->strides[0] +
                       (col + j) * output->strides[1] + y * output->strides[2] + x * output->strides[3]] =
              (float)result[i * 16 + j] * (inputScale * weightScale) +
              bias->data[bias->offset + (col + j) * bias->strides[0]];
        }
    }
  bb_mem_release(0); bb_mem_release(1); bb_mem_release(2); bb_mem_release(3);
  free(panels);
}

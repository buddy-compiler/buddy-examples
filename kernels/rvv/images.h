#pragma once
// The RVV kernel images built by stack/kernels/rvv, embedded from the build
// directory given with --embed-dir. Each translation unit keeps only the
// images it uses, so libraries built for different targets link together.
#include <cstdint>

namespace images {
struct KernelImage {
  const uint8_t *bytes;
  uint32_t size;
  uint32_t entry;
  uint32_t text_bytes;
};

namespace {
// Header words, see kernel.ld: 1 = code bytes, 2 = entry.
constexpr uint32_t word(const uint8_t *image, int index) {
  return image[4 * index] | image[4 * index + 1] << 8 |
         image[4 * index + 2] << 16 | uint32_t(image[4 * index + 3]) << 24;
}

alignas(4) constexpr uint8_t silu_image[] = {
#embed "silu.image"
};
constexpr KernelImage silu{silu_image, sizeof(silu_image), word(silu_image, 2),
                           word(silu_image, 1)};
alignas(4) constexpr uint8_t swiglu_image[] = {
#embed "swiglu.image"
};
constexpr KernelImage swiglu{swiglu_image, sizeof(swiglu_image),
                             word(swiglu_image, 2), word(swiglu_image, 1)};
alignas(4) constexpr uint8_t snake_image[] = {
#embed "snake.image"
};
constexpr KernelImage snake{snake_image, sizeof(snake_image),
                            word(snake_image, 2), word(snake_image, 1)};
alignas(4) constexpr uint8_t quant_image[] = {
#embed "quant.image"
};
constexpr KernelImage quant{quant_image, sizeof(quant_image),
                            word(quant_image, 2), word(quant_image, 1)};
alignas(4) constexpr uint8_t cache_quant_image[] = {
#embed "cache_quant.image"
};
constexpr KernelImage cache_quant{cache_quant_image, sizeof(cache_quant_image),
                                  word(cache_quant_image, 2),
                                  word(cache_quant_image, 1)};
alignas(4) constexpr uint8_t dequant_image[] = {
#embed "dequant.image"
};
constexpr KernelImage dequant{dequant_image, sizeof(dequant_image),
                              word(dequant_image, 2), word(dequant_image, 1)};
alignas(4) constexpr uint8_t matmul_image[] = {
#embed "matmul.image"
};
constexpr KernelImage matmul{matmul_image, sizeof(matmul_image),
                             word(matmul_image, 2), word(matmul_image, 1)};
alignas(4) constexpr uint8_t norm_image[] = {
#embed "norm.image"
};
constexpr KernelImage norm{norm_image, sizeof(norm_image), word(norm_image, 2),
                           word(norm_image, 1)};
alignas(4) constexpr uint8_t norm_reduce_image[] = {
#embed "norm_reduce.image"
};
constexpr KernelImage norm_reduce{norm_reduce_image, sizeof(norm_reduce_image),
                                  word(norm_reduce_image, 2),
                                  word(norm_reduce_image, 1)};
alignas(4) constexpr uint8_t norm_apply_image[] = {
#embed "norm_apply.image"
};
constexpr KernelImage norm_apply{norm_apply_image, sizeof(norm_apply_image),
                                 word(norm_apply_image, 2),
                                 word(norm_apply_image, 1)};
alignas(4) constexpr uint8_t norm_no_weight_image[] = {
#embed "norm_no_weight.image"
};
constexpr KernelImage norm_no_weight{
    norm_no_weight_image, sizeof(norm_no_weight_image),
    word(norm_no_weight_image, 2), word(norm_no_weight_image, 1)};
alignas(4) constexpr uint8_t softmax_image[] = {
#embed "softmax.image"
};
constexpr KernelImage softmax{softmax_image, sizeof(softmax_image),
                              word(softmax_image, 2), word(softmax_image, 1)};
alignas(4) constexpr uint8_t attention_softmax_image[] = {
#embed "attention_softmax.image"
};
constexpr KernelImage attention_softmax{
    attention_softmax_image, sizeof(attention_softmax_image),
    word(attention_softmax_image, 2), word(attention_softmax_image, 1)};
alignas(4) constexpr uint8_t logsumexp_softmax_image[] = {
#embed "logsumexp_softmax.image"
};
constexpr KernelImage logsumexp_softmax{
    logsumexp_softmax_image, sizeof(logsumexp_softmax_image),
    word(logsumexp_softmax_image, 2), word(logsumexp_softmax_image, 1)};
alignas(4) constexpr uint8_t rope_image[] = {
#embed "rope.image"
};
constexpr KernelImage rope{rope_image, sizeof(rope_image), word(rope_image, 2),
                           word(rope_image, 1)};
alignas(4) constexpr uint8_t layernorm_image[] = {
#embed "layernorm.image"
};
constexpr KernelImage layernorm{layernorm_image, sizeof(layernorm_image),
                                word(layernorm_image, 2),
                                word(layernorm_image, 1)};
alignas(4) constexpr uint8_t gelu_image[] = {
#embed "gelu.image"
};
constexpr KernelImage gelu{gelu_image, sizeof(gelu_image), word(gelu_image, 2),
                           word(gelu_image, 1)};
alignas(4) constexpr uint8_t tanh_image[] = {
#embed "tanh.image"
};
constexpr KernelImage tanh{tanh_image, sizeof(tanh_image), word(tanh_image, 2),
                           word(tanh_image, 1)};
alignas(4) constexpr uint8_t sin_image[] = {
#embed "sin.image"
};
constexpr KernelImage sin{sin_image, sizeof(sin_image), word(sin_image, 2),
                          word(sin_image, 1)};
alignas(4) constexpr uint8_t cos_image[] = {
#embed "cos.image"
};
constexpr KernelImage cos{cos_image, sizeof(cos_image), word(cos_image, 2),
                          word(cos_image, 1)};
alignas(4) constexpr uint8_t flash_attention_image[] = {
#embed "flash_attention.image"
};
constexpr KernelImage flash_attention{
    flash_attention_image, sizeof(flash_attention_image),
    word(flash_attention_image, 2), word(flash_attention_image, 1)};
alignas(4) constexpr uint8_t flash_attention_mxfp8_image[] = {
#embed "flash_attention_mxfp8.image"
};
constexpr KernelImage flash_attention_mxfp8{
    flash_attention_mxfp8_image, sizeof(flash_attention_mxfp8_image),
    word(flash_attention_mxfp8_image, 2), word(flash_attention_mxfp8_image, 1)};
alignas(4) constexpr uint8_t pointwise_image[] = {
#embed "pointwise.image"
};
constexpr KernelImage pointwise{pointwise_image, sizeof(pointwise_image),
                                word(pointwise_image, 2),
                                word(pointwise_image, 1)};
} // namespace
} // namespace images

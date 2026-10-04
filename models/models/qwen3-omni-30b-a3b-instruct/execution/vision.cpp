#include "vision.h"
#include <cstdlib>
#include <fstream>
#include <runtime.h>

using Indices = MemRef<int64_t, 2>;
using Mask = MemRef<bool, 2>;
using PatchKernel = void (*)(Matrix *, Floats *, Bytes *, Matrix *, Indices *,
                             Matrix *);
using AttentionKernel = void (*)(Matrix *, Floats *, Bytes *, Matrix *,
                                 Matrix *, Matrix *, Mask *);
using MatrixKernel = void (*)(Matrix *, Floats *, Bytes *, Matrix *);

#define VISION_KERNELS(N)                                                      \
  extern "C" void _mlir_ciface_forward_vision_patch_##N(                       \
      Matrix *, Floats *, Bytes *, Matrix *, Indices *, Matrix *);             \
  extern "C" void _mlir_ciface_forward_vision_attention_##N(                   \
      Matrix *, Floats *, Bytes *, Matrix *, Matrix *, Matrix *, Mask *);      \
  extern "C" void _mlir_ciface_forward_vision_mlp_##N(Matrix *, Floats *,      \
                                                      Bytes *, Matrix *);      \
  extern "C" void _mlir_ciface_forward_vision_merge_##N(Matrix *, Floats *,    \
                                                        Bytes *, Matrix *);    \
  extern "C" void _mlir_ciface_forward_vision_deepstack_##N(                   \
      Matrix *, Floats *, Bytes *, Matrix *);
VISION_KERNELS(4)
VISION_KERNELS(16)
VISION_KERNELS(64)
VISION_KERNELS(256)
#undef VISION_KERNELS

static constexpr PatchKernel patches[] = {
    _mlir_ciface_forward_vision_patch_4, _mlir_ciface_forward_vision_patch_16,
    _mlir_ciface_forward_vision_patch_64,
    _mlir_ciface_forward_vision_patch_256};
static constexpr AttentionKernel attention[] = {
    _mlir_ciface_forward_vision_attention_4,
    _mlir_ciface_forward_vision_attention_16,
    _mlir_ciface_forward_vision_attention_64,
    _mlir_ciface_forward_vision_attention_256};
static constexpr MatrixKernel mlp[] = {
    _mlir_ciface_forward_vision_mlp_4, _mlir_ciface_forward_vision_mlp_16,
    _mlir_ciface_forward_vision_mlp_64, _mlir_ciface_forward_vision_mlp_256};
static constexpr MatrixKernel merge[] = {_mlir_ciface_forward_vision_merge_4,
                                         _mlir_ciface_forward_vision_merge_16,
                                         _mlir_ciface_forward_vision_merge_64,
                                         _mlir_ciface_forward_vision_merge_256};
static constexpr MatrixKernel deepstack[] = {
    _mlir_ciface_forward_vision_deepstack_4,
    _mlir_ciface_forward_vision_deepstack_16,
    _mlir_ciface_forward_vision_deepstack_64,
    _mlir_ciface_forward_vision_deepstack_256};

Vision::Vision(const std::filesystem::path &directory) {
  std::ifstream input;
  input.exceptions(std::ios::failbit | std::ios::badbit);
  input.open(directory / "vision-layout.bin", std::ios::binary);
  std::array<uint64_t, 4> header;
  input.read(reinterpret_cast<char *>(header.data()), sizeof(header));
  if (header[0] != 0x564953490001 || header[3] != 2 * visionLayers + 5)
    throw std::runtime_error(
        "vision weight layout does not match compiled encoder");
  floats.resize(header[1]);
  bytes.resize(header[2]);
  regions.resize(header[3]);
  input.read(reinterpret_cast<char *>(regions.data()),
             regions.size() * sizeof(Region));
  input.close();
  input.open(directory / "vision.f32", std::ios::binary);
  input.read(reinterpret_cast<char *>(floats.data()),
             floats.size() * sizeof(float));
  input.close();
  input.open(directory / "vision.bin", std::ios::binary);
  input.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
  runtime_init(1024 * 1024);
  workspace = aligned_alloc(64, 256 * 1024 * 1024);
  if (!workspace)
    throw std::bad_alloc();
  workspace_init(workspace, 256 * 1024 * 1024);
  std::cerr << "Vision ready: layers=" << visionLayers
            << " weight_bytes=" << floats.size() * 4 + bytes.size() << '\n';
}

Vision::~Vision() { free(workspace); }

void Vision::execute(const Command &command) {
  auto [operation, count, unused0, unused1] = command;
  if (operation != 12 || !count || count % visionMerge ||
      count > visionBuckets[3] || unused0 || unused1)
    throw std::runtime_error("invalid vision encoder command");
  size_t bucket = 0;
  while (visionBuckets[bucket] < count)
    ++bucket;
  size_t length = visionBuckets[bucket];
  Matrix input({length, visionPatchWidth}, 0.0f),
      coefficients({length, 4}, 0.0f);
  Indices indices({length, 4}, int64_t(0));
  Matrix cosine({length, visionHeadDim}, 1.0f),
      sine({length, visionHeadDim}, 0.0f);
  Mask mask({length, length}, false);
  read_values(input.getData(), count * visionPatchWidth);
  read_values(indices.getData(), count * 4);
  read_values(coefficients.getData(), count * 4);
  read_values(cosine.getData(), count * visionHeadDim);
  read_values(sine.getData(), count * visionHeadDim);
  for (size_t row = 0; row < count; ++row)
    read_values(mask.getData() + row * length, count);
  for (size_t row = 0; row < count; ++row)
    for (size_t col = count; col < length; ++col)
      mask[row * length + col] = true;
  workspace_begin(workspace, 256 * 1024 * 1024);
  Matrix hidden({length, visionWidth}, false, 0);
  auto parameters = [&](size_t index) {
    const auto &region = regions.at(index);
    return std::pair{
        View<int8_t, 1>(bytes.data() + region.byte_offset, {region.byte_count}),
        View<float, 1>(floats.data() + region.float_offset,
                       {region.float_count})};
  };
  auto [packed, fp] = parameters(0);
  patches[bucket](&hidden, &fp, &packed, &input, &indices, &coefficients);
  std::vector<std::vector<float>> features;
  for (size_t layer = 0; layer < visionLayers; ++layer) {
    auto [ap, af] = parameters(1 + 2 * layer);
    Matrix attended({length, visionWidth}, false, 0);
    attention[bucket](&attended, &af, &ap, &hidden, &cosine, &sine, &mask);
    workspace_free(hidden.release());
    auto [mp, mf] = parameters(2 + 2 * layer);
    mlp[bucket](&hidden, &mf, &mp, &attended);
    workspace_free(attended.release());
    for (size_t index = 0; index < 3; ++index) {
      if (layer != visionDeepstackLayers[index])
        continue;
      auto [dp, df] = parameters(2 * visionLayers + 2 + index);
      Matrix feature({length / visionMerge, visionOutputWidth}, false, 0);
      deepstack[bucket](&feature, &df, &dp, &hidden);
      features.emplace_back(feature.getData(),
                            feature.getData() +
                                count / visionMerge * visionOutputWidth);
      workspace_free(feature.release());
    }
  }
  auto [ep, ef] = parameters(2 * visionLayers + 1);
  Matrix result({length / visionMerge, visionOutputWidth}, false, 0);
  merge[bucket](&result, &ef, &ep, &hidden);
  write_values(result.getData(), count / visionMerge * visionOutputWidth);
  for (const auto &feature : features)
    write_values(feature.data(), feature.size());
  workspace_free(result.release());
  workspace_free(hidden.release());
}

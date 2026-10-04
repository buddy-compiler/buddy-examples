#include "audio.h"
#include <cstdlib>
#include <fstream>
#include <runtime.h>

using AttentionKernel = void (*)(Matrix *, Floats *, Bytes *, Matrix *,
                                 MemRef<bool, 2> *);
using MatrixKernel = void (*)(Matrix *, Floats *, Bytes *, Matrix *);
#define AUDIO_KERNELS(N)                                                       \
  extern "C" void _mlir_ciface_forward_audio_attention_##N(                    \
      Matrix *, Floats *, Bytes *, Matrix *, MemRef<bool, 2> *);               \
  extern "C" void _mlir_ciface_forward_audio_mlp_##N(Matrix *, Floats *,       \
                                                     Bytes *, Matrix *);       \
  extern "C" void _mlir_ciface_forward_audio_output_##N(Matrix *, Floats *,    \
                                                        Bytes *, Matrix *);
AUDIO_KERNELS(1)
AUDIO_KERNELS(4)
AUDIO_KERNELS(16)
AUDIO_KERNELS(64)
AUDIO_KERNELS(128)
#undef AUDIO_KERNELS
static constexpr AttentionKernel attention[] = {
    _mlir_ciface_forward_audio_attention_1,
    _mlir_ciface_forward_audio_attention_4,
    _mlir_ciface_forward_audio_attention_16,
    _mlir_ciface_forward_audio_attention_64,
    _mlir_ciface_forward_audio_attention_128};
static constexpr MatrixKernel mlp[] = {
    _mlir_ciface_forward_audio_mlp_1, _mlir_ciface_forward_audio_mlp_4,
    _mlir_ciface_forward_audio_mlp_16, _mlir_ciface_forward_audio_mlp_64,
    _mlir_ciface_forward_audio_mlp_128};
static constexpr MatrixKernel output[] = {
    _mlir_ciface_forward_audio_output_1, _mlir_ciface_forward_audio_output_4,
    _mlir_ciface_forward_audio_output_16, _mlir_ciface_forward_audio_output_64,
    _mlir_ciface_forward_audio_output_128};

Audio::Audio(const std::filesystem::path &directory) {
  std::ifstream input;
  input.exceptions(std::ios::failbit | std::ios::badbit);
  input.open(directory / "audio-layout.bin", std::ios::binary);
  std::array<uint64_t, 4> header;
  input.read(reinterpret_cast<char *>(header.data()), sizeof(header));
  if (header[0] != 0x415544490001 || header[3] != 2 * audioLayers + 5)
    throw std::runtime_error(
        "audio weight layout does not match compiled encoder");
  floats.resize(header[1]);
  bytes.resize(header[2]);
  regions.resize(header[3]);
  input.read(reinterpret_cast<char *>(regions.data()),
             regions.size() * sizeof(Region));
  input.close();
  input.open(directory / "audio.f32", std::ios::binary);
  input.read(reinterpret_cast<char *>(floats.data()),
             floats.size() * sizeof(float));
  input.close();
  input.open(directory / "audio.bin", std::ios::binary);
  input.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
  runtime_init(1024 * 1024);
  workspace = aligned_alloc(64, 256 * 1024 * 1024);
  if (!workspace)
    throw std::bad_alloc();
  workspace_init(workspace, 256 * 1024 * 1024);
  std::cerr << "Audio ready: layers=" << audioLayers
            << " weight_bytes=" << floats.size() * 4 + bytes.size() << '\n';
}
Audio::~Audio() { free(workspace); }

std::pair<View<float, 1>, View<int8_t, 1>> Audio::parameters(size_t index) {
  const auto &region = regions.at(index);
  return {
      View<float, 1>(floats.data() + region.float_offset, {region.float_count}),
      View<int8_t, 1>(bytes.data() + region.byte_offset, {region.byte_count})};
}

void Audio::execute(const Command &command) {
  auto [operation, frames, unused0, unused1] = command;
  if (operation != 14 || !frames || unused0 || unused1)
    throw std::runtime_error("invalid audio encoder command");
  std::vector<float> input(frames * 128);
  read_values(input.data(), input.size());
  workspace_begin(workspace, 256 * 1024 * 1024);
  auto projected = downsample(input, frames);
  size_t total = projected.size() / audioWidth;
  size_t window = ((std::min<size_t>(frames, 100) + 7) / 8) * 8;
  for (size_t begin = 0; begin < total; begin += window) {
    size_t count = std::min(window, total - begin), bucket = 0;
    while (audioBuckets[bucket] < count)
      ++bucket;
    size_t length = audioBuckets[bucket];
    Matrix hidden({length, audioWidth}, 0.0f);
    std::copy_n(projected.data() + begin * audioWidth, count * audioWidth,
                hidden.getData());
    MemRef<bool, 2> mask({length, length}, false);
    for (size_t row = 0; row < count; ++row)
      for (size_t col = count; col < length; ++col)
        mask[row * length + col] = true;
    bool allocated = false;
    for (size_t layer = 0; layer < audioLayers; ++layer) {
      auto [af, ap] = parameters(4 + 2 * layer);
      Matrix attended({length, audioWidth}, false, 0);
      attention[bucket](&attended, &af, &ap, &hidden, &mask);
      if (allocated)
        workspace_free(hidden.release());
      auto [mf, mp] = parameters(5 + 2 * layer);
      Matrix result({length, audioWidth}, false, 0);
      mlp[bucket](&result, &mf, &mp, &attended);
      workspace_free(attended.release());
      hidden = std::move(result);
      allocated = true;
    }
    auto [fp, packed] = parameters(4 + 2 * audioLayers);
    Matrix result({length, audioOutputWidth}, false, 0);
    output[bucket](&result, &fp, &packed, &hidden);
    write_values(result.getData(), count * audioOutputWidth);
    workspace_free(result.release());
    workspace_free(hidden.release());
  }
}

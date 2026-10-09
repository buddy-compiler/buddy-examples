#include "audio.h"
#include <runtime.h>

extern "C" void _mlir_ciface_forward_audio_conv1(Cache *, Floats *, Bytes *,
                                                 Cache *, Slots *);
extern "C" void _mlir_ciface_forward_audio_conv2(Cache *, Floats *, Bytes *,
                                                 Cache *, Slots *);
extern "C" void _mlir_ciface_forward_audio_conv3(Cache *, Floats *, Bytes *,
                                                 Cache *, Slots *);
extern "C" void _mlir_ciface_forward_audio_projection(Matrix *, Floats *,
                                                      Bytes *, Cache *);

std::vector<float> Audio::downsample(const std::vector<float> &input,
                                     size_t frames) {
  std::vector<float> projected;
  for (size_t offset = 0; offset < frames; offset += 100) {
    size_t count = std::min<size_t>(100, frames - offset);
    Cache features({1, 1, 128, 100}, 0.0f), conv1({1, 480, 64, 50}, false, 0),
        conv2({1, 480, 32, 25}, false, 0), conv3({1, 480, 16, 13}, false, 0);
    for (size_t mel = 0; mel < 128; ++mel)
      for (size_t time = 0; time < count; ++time)
        features[mel * 100 + time] = input[(offset + time) * 128 + mel];
    int64_t valid_frames = std::min<size_t>(frames, 100);
    View<int64_t, 1> valid(&valid_frames, {1});
    auto [f1, p1] = parameters(0);
    _mlir_ciface_forward_audio_conv1(&conv1, &f1, &p1, &features, &valid);
    valid_frames = (valid_frames + 1) / 2;
    auto [f2, p2] = parameters(1);
    _mlir_ciface_forward_audio_conv2(&conv2, &f2, &p2, &conv1, &valid);
    workspace_free(conv1.release());
    valid_frames = (valid_frames + 1) / 2;
    auto [f3, p3] = parameters(2);
    _mlir_ciface_forward_audio_conv3(&conv3, &f3, &p3, &conv2, &valid);
    workspace_free(conv2.release());
    auto [fp, packed] = parameters(3);
    Matrix result({13, audioWidth}, false, 0);
    _mlir_ciface_forward_audio_projection(&result, &fp, &packed, &conv3);
    workspace_free(conv3.release());
    projected.insert(projected.end(), result.getData(),
                     result.getData() + (count + 7) / 8 * audioWidth);
    workspace_free(result.release());
  }
  return projected;
}

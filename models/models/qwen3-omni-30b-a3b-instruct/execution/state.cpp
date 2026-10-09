#include "state.h"
#include <runtime.h>
#include <fstream>
#include <cstdlib>

Thinker::Thinker(const std::filesystem::path &directory) {
  std::ifstream layout;
  layout.exceptions(std::ios::failbit | std::ios::badbit);
  layout.open(directory / "layout.bin", std::ios::binary);
  std::array<uint64_t, 8> header;
  layout.read(reinterpret_cast<char *>(header.data()), sizeof(header));
  if (header[0] != 0x4F4D4E490001 || header[4] != parts || header[1] >= header[2] ||
      header[2] > layers || header[3] >= parts || header[7] != 3 + (header[2] - header[1]) * (expertsCount + 2))
    throw std::runtime_error("weight layout does not match compiled Thinker");
  first = header[1]; last = header[2]; rank = header[3];
  regions.resize(header[7]);
  layout.read(reinterpret_cast<char *>(regions.data()), regions.size() * sizeof(Region));
  float_count = header[5]; byte_count = header[6];
  floats.reset(new float[float_count]); bytes.reset(new int8_t[byte_count]);
  std::ifstream input;
  input.exceptions(std::ios::failbit | std::ios::badbit);
  input.open(directory / "params.f32", std::ios::binary);
  input.read(reinterpret_cast<char *>(floats.get()), float_count * sizeof(float));
  input.close();
  input.open(directory / "weights.bin", std::ios::binary);
  input.read(reinterpret_cast<char *>(bytes.get()), byte_count);
  const std::vector<size_t> shape{1, kvHeads, cacheLength, headSize};
  for (size_t layer = first; layer < last; ++layer) {
    keys.emplace_back(shape, 0.0f); values.emplace_back(shape, 0.0f);
  }
  cache_tokens.resize(last - first, 0);
  runtime_init(1024 * 1024);
  workspace = aligned_alloc(64, workspaceBytes);
  if (!workspace) throw std::bad_alloc();
  workspace_init(workspace, workspaceBytes);
  std::cerr << "Thinker ready: layers=" << first << ':' << last << " rank=" << rank
            << " weight_bytes=" << float_count * 4 + byte_count << '\n';
}

Thinker::~Thinker() { free(workspace); }

View<float, 1> Thinker::float_parameters(size_t index) {
  const auto &region = regions.at(index);
  if (!region.float_count) throw std::runtime_error("FP32 parameters are not resident on this tile");
  return {floats.get() + region.float_offset, {region.float_count}};
}

View<int8_t, 1> Thinker::byte_parameters(size_t index) {
  const auto &region = regions.at(index);
  if (!region.byte_count) throw std::runtime_error("MXFP8 parameters are not resident on this tile");
  return {bytes.get() + region.byte_offset, {region.byte_count}};
}

size_t Thinker::layer_region(size_t layer) {
  if (layer < first || layer >= last) throw std::runtime_error("layer belongs to another chip");
  return 3 + (layer - first) * (expertsCount + 2);
}

void Thinker::execute(const Command &command) {
  const auto [operation, count, start, layer] = command;
  if (!count || count > prefillLength) throw std::runtime_error("token count exceeds compiled capacity");
  workspace_begin(workspace, workspaceBytes);
  switch (operation) {
  case 1: embedding(count); break;
  case 2: attention(count, start, layer); break;
  case 3: router(count, layer); break;
  case 4: experts(count, layer); break;
  case 5:
    if (start > 1) throw std::runtime_error("invalid residual reduction flag");
    reduce(count, start == 1); break;
  case 6:
    if (count != 1) throw std::runtime_error("output projection consumes one token");
    output(); break;
  case 7: norm(count); break;
  case 9: send(count, start, layer); break;
  case 10: receive(count, start); break;
  default: throw std::runtime_error("unknown Thinker command");
  }
}

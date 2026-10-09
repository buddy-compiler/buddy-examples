#include "talker.h"
#include <cstdlib>
#include <fstream>
#include <runtime.h>

using namespace TalkerParams;

Talker::Talker(const std::filesystem::path &directory) {
  std::ifstream input;
  input.exceptions(std::ios::failbit | std::ios::badbit);
  input.open(directory / "talker-layout.bin", std::ios::binary);
  std::array<uint64_t, 4> header;
  input.read(reinterpret_cast<char *>(header.data()), sizeof(header));
  if (header[0] != 0x54414C4B0001 ||
      header[3] !=
          2 * groups + 4 + layerCount * (expertCount + 3) + 2 * predictorLayers)
    throw std::runtime_error(
        "Talker weight layout does not match compiled model");
  floats.reset(new float[header[1]]);
  bytes.reset(new int8_t[header[2]]);
  regions.resize(header[3]);
  input.read(reinterpret_cast<char *>(regions.data()),
             regions.size() * sizeof(Region));
  input.close();
  input.open(directory / "talker.f32", std::ios::binary);
  input.read(reinterpret_cast<char *>(floats.get()), header[1] * sizeof(float));
  input.close();
  input.open(directory / "talker.bin", std::ios::binary);
  input.read(reinterpret_cast<char *>(bytes.get()), header[2]);
  for (size_t layer = 0; layer < layerCount; ++layer) {
    keys.emplace_back(std::vector<size_t>{1, kvHeads, capacity, headDim}, 0.0f);
    values.emplace_back(std::vector<size_t>{1, kvHeads, capacity, headDim},
                        0.0f);
  }
  for (size_t layer = 0; layer < predictorLayers; ++layer) {
    predictor_keys.emplace_back(
        std::vector<size_t>{1, predictorKVHeads, predictorCapacity, headDim},
        0.0f);
    predictor_values.emplace_back(
        std::vector<size_t>{1, predictorKVHeads, predictorCapacity, headDim},
        0.0f);
  }
  runtime_init(1024 * 1024);
  workspace = aligned_alloc(64, 64 * 1024 * 1024);
  if (!workspace)
    throw std::bad_alloc();
  workspace_init(workspace, 64 * 1024 * 1024);
  std::cerr << "Talker ready: layers=" << layerCount
            << " predictor_layers=" << predictorLayers
            << " weight_bytes=" << header[1] * 4 + header[2] << '\n';
}
Talker::~Talker() { free(workspace); }

std::pair<View<float, 1>, View<int8_t, 1>> Talker::parameters(size_t index) {
  const auto &region = regions.at(index);
  return {
      View<float, 1>(floats.get() + region.float_offset, {region.float_count}),
      View<int8_t, 1>(bytes.get() + region.byte_offset, {region.byte_count})};
}

void Talker::execute(const Command &command) {
  auto [operation, count, start, auxiliary] = command;
  if (!count || count > prefill || auxiliary)
    throw std::runtime_error("Talker token count exceeds compiled capacity");
  workspace_begin(workspace, 64 * 1024 * 1024);
  switch (operation) {
  case 16:
    resize(count, start);
    break;
  case 17:
    embedding(count, start);
    break;
  case 18:
    forward(count, start, false);
    break;
  case 19:
    if (count > 2)
      throw std::runtime_error("code predictor consumes at most two tokens");
    forward(count, start, true);
    break;
  case 20:
    if (count != 1)
      throw std::runtime_error("Talker logits consume one token");
    logits(0, false);
    break;
  case 21:
    if (count != 1 || start >= groups - 1)
      throw std::runtime_error("invalid predictor output head");
    logits(start, true);
    break;
  default:
    throw std::runtime_error("unknown Talker command");
  }
}

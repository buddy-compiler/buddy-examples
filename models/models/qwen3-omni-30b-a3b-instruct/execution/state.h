#pragma once
#include "view.h"
#include "io.h"
#include "model-parameters.h"
#include <filesystem>
#include <memory>
#include <vector>

struct Region { uint64_t float_offset, float_count, byte_offset, byte_count; };

class Thinker {
public:
  explicit Thinker(const std::filesystem::path &directory);
  ~Thinker();
  void execute(const Command &command);
private:
  size_t first, last, rank;
  std::unique_ptr<float[]> floats;
  std::unique_ptr<int8_t[]> bytes;
  size_t float_count, byte_count;
  std::vector<Region> regions;
  std::vector<Cache> keys, values;
  std::vector<size_t> cache_tokens;
  void *workspace;
  std::vector<float> exchange;
  View<float, 1> float_parameters(size_t index);
  View<int8_t, 1> byte_parameters(size_t index);
  size_t layer_region(size_t layer);
  void embedding(size_t count);
  void attention(size_t count, size_t start, size_t layer);
  void router(size_t count, size_t layer);
  void experts(size_t count, size_t layer);
  void reduce(size_t count, bool residual);
  void norm(size_t count);
  void output();
  void send(size_t count, size_t chip, uint64_t destination);
  void receive(size_t count, size_t source);
};

#pragma once
#include "io.h"
#include "talker-parameters.h"
#include "view.h"
#include <filesystem>
#include <memory>
#include <vector>

class Talker {
public:
  explicit Talker(const std::filesystem::path &directory);
  ~Talker();
  void execute(const Command &command);

private:
  std::unique_ptr<float[]> floats;
  std::unique_ptr<int8_t[]> bytes;
  std::vector<Region> regions;
  std::vector<Cache> keys, values, predictor_keys, predictor_values;
  size_t position = 0, predictor_position = 0;
  void *workspace;
  std::pair<View<float, 1>, View<int8_t, 1>> parameters(size_t index);
  void embedding(size_t count, size_t group);
  void resize(size_t count, size_t kind);
  void forward(size_t count, size_t start, bool predictor);
  void experts(Matrix &hidden, size_t count, size_t layer);
  void logits(size_t group, bool predictor);
};

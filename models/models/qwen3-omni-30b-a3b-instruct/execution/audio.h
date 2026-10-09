#pragma once
#include "audio-parameters.h"
#include "io.h"
#include "view.h"
#include <filesystem>
#include <vector>

class Audio {
public:
  explicit Audio(const std::filesystem::path &directory);
  ~Audio();
  void execute(const Command &command);

private:
  std::vector<float> floats;
  std::vector<int8_t> bytes;
  std::vector<Region> regions;
  void *workspace;
  std::pair<View<float, 1>, View<int8_t, 1>> parameters(size_t index);
  std::vector<float> downsample(const std::vector<float> &input, size_t frames);
};

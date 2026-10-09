#pragma once
#include "io.h"
#include "view.h"
#include "wave-parameters.h"
#include <filesystem>
#include <memory>
#include <vector>

class Wave {
public:
  explicit Wave(const std::filesystem::path &directory);
  ~Wave();
  void execute(const Command &command);

private:
  std::unique_ptr<float[]> floats;
  std::unique_ptr<int8_t[]> bytes;
  std::vector<Region> regions;
  void *workspace;
  std::pair<View<float, 1>, View<int8_t, 1>> parameters(size_t index);
};

#pragma once
#include "io.h"
#include "view.h"
#include "vision-parameters.h"
#include <filesystem>
#include <vector>

class Vision {
public:
  explicit Vision(const std::filesystem::path &directory);
  ~Vision();
  void execute(const Command &command);

private:
  std::vector<float> floats;
  std::vector<int8_t> bytes;
  std::vector<Region> regions;
  void *workspace;
};

#include "execution.h"
#include "generation.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sched.h>
#include <stdexcept>
#include <system_error>
#include <time.h>

int main(int argc, char **argv) {
  if (argc != 3 && argc != 4)
    throw std::runtime_error(
        "usage: qwen-run MODEL_DIRECTORY INPUT_RESOURCE [RESOURCE_INDEX]");
  const std::filesystem::path directory = std::filesystem::absolute(argv[1]);
  const std::filesystem::path inputPath = directory / argv[2];
  resources::configure(directory, argc == 4 ? argv[3] : nullptr);
  const size_t inputBytes =
      resources::indexed
          ? resources::index.entries
                .at(inputPath.lexically_normal()
                        .lexically_relative(resources::index.root)
                        .generic_string())
                .size
          : std::filesystem::file_size(inputPath);
  resources::Mapping input(inputPath, inputBytes, alignof(uint64_t));
  if (inputBytes < 40)
    throw std::runtime_error("incomplete Qwen input header");
  const uint64_t *header = static_cast<const uint64_t *>(input.data());
  const size_t tokenCount = header[0], maxTokens = header[1],
               eosCount = header[2], tileCount = header[3];
  double temperature;
  std::memcpy(&temperature, &header[4], sizeof(temperature));
  const auto &shape = modelShape();
  if (!tokenCount || tokenCount > shape.prefill || !maxTokens ||
      maxTokens - 1 > shape.cache - tokenCount || tileCount != shape.tiles ||
      !std::isfinite(temperature) || temperature < 0 ||
      inputBytes != 40 + (tokenCount + eosCount + tileCount) * sizeof(uint64_t))
    throw std::runtime_error("invalid Qwen input parameters");
  const int64_t *tokenIds = reinterpret_cast<const int64_t *>(header + 5);
  const int64_t *eosIds = tokenIds + tokenCount;
  const uint64_t *tileIds =
      reinterpret_cast<const uint64_t *>(eosIds + eosCount);
  for (size_t index = 0; index < tokenCount + eosCount; ++index)
    if (tokenIds[index] < 0 || size_t(tokenIds[index]) >= shape.vocabulary)
      throw std::runtime_error("input token ID exceeds vocabulary");
  std::vector<uint64_t> tiles(tileIds, tileIds + tileCount);
  for (size_t rank = 0; rank < tiles.size(); ++rank)
    if (tiles[rank] == 0 ||
        std::count(tiles.begin(), tiles.end(), tiles[rank]) != 1)
      throw std::runtime_error("duplicate or main tile in compute plan");
  cpu_set_t cpus;
  CPU_ZERO(&cpus);
  CPU_SET(controlCpu(0), &cpus);
  if (sched_setaffinity(0, sizeof(cpus), &cpus))
    throw std::system_error(errno, std::generic_category(),
                            "pin model main CPU");
  Execution execution(directory, tiles);
  timespec before, after;
  clock_gettime(CLOCK_MONOTONIC, &before);
  auto generated = generate(
      execution, std::vector<int64_t>(tokenIds, tokenIds + tokenCount),
      std::vector<int64_t>(eosIds, eosIds + eosCount), maxTokens, temperature);
  execution.close();
  clock_gettime(CLOCK_MONOTONIC, &after);
  double elapsed = double(after.tv_sec - before.tv_sec) +
                   double(after.tv_nsec - before.tv_nsec) * 1e-9;
  std::cout << "{\"token_ids\":[";
  for (size_t index = 0; index < generated.size(); ++index) {
    if (index)
      std::cout << ',';
    std::cout << generated[index];
  }
  std::cout << "],\"guest_wall_seconds\":" << elapsed << "}" << std::endl;
}

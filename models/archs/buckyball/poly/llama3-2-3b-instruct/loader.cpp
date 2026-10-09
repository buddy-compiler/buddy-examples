#include "execution_state.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <resources.h>
#include <sched.h>
#include <stdexcept>
#include <system_error>
#include <topology.h>

int controlCpu(size_t tile) {
  for (uint32_t hart = 0; hart < BB_HART_NUM; ++hart) {
    const auto id = bb_topology_core_id(hart);
    if (id.tile == tile && id.core == 0)
      return int(hart);
  }
  throw std::runtime_error("tile has no control CPU");
}
Execution::Execution(const std::filesystem::path &directory,
                     const std::filesystem::path &inputResource,
                     const char *index) {
  const auto inputPath = directory / inputResource;
  resources::configure(directory, index);
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
    throw std::runtime_error("incomplete Llama input header");
  const auto *header = static_cast<const uint64_t *>(input.data());
  const size_t tokenCount = header[0], maxTokens = header[1],
               eosCount = header[2], tileCount = header[3];
  double temperature;
  std::memcpy(&temperature, &header[4], sizeof(temperature));
  const auto &shape = modelShape();
  if (!tokenCount || tokenCount > shape.prefill || !maxTokens ||
      maxTokens - 1 > shape.cache - tokenCount || tileCount != shape.tiles ||
      !std::isfinite(temperature) || temperature < 0 ||
      inputBytes != 40 + (tokenCount + eosCount + tileCount) * sizeof(uint64_t))
    throw std::runtime_error("invalid Llama input parameters");
  const auto *tokenIds = reinterpret_cast<const int64_t *>(header + 5);
  const auto *eosIds = tokenIds + tokenCount;
  const auto *tileIds = reinterpret_cast<const uint64_t *>(eosIds + eosCount);
  for (size_t i = 0; i < tokenCount + eosCount; ++i)
    if (tokenIds[i] < 0 || size_t(tokenIds[i]) >= shape.vocabulary)
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
  Request request{{tokenIds, tokenIds + tokenCount},
                  {eosIds, eosIds + eosCount},
                  maxTokens,
                  temperature};
  state = std::make_unique<State>(directory, tiles);
  state->request = std::move(request);
}

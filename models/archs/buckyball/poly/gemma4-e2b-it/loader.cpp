#include "execution_state.h"
#include "gemma-parameters.h"
#include "gemma-plan.h"
#include <cstring>
#include <resources.h>
#include <sched.h>
#include <stdexcept>
#include <system_error>
#include <topology.h>

Execution::Execution(const std::filesystem::path &directory,
                     const std::filesystem::path &inputResource,
                     const char *index)
    : state(std::make_unique<State>()) {
  resources::configure(directory, index);
  const auto inputPath = directory / inputResource;
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
    throw std::runtime_error("incomplete Gemma input header");
  const auto *header = static_cast<const uint64_t *>(input.data());
  const size_t tokenCount = header[0], maxTokens = header[1],
               eosCount = header[2], tileCount = header[3];
  double temperature;
  std::memcpy(&temperature, &header[4], sizeof(temperature));
  if (!tokenCount || tokenCount > PrefillLength || !maxTokens ||
      maxTokens > 1 + CacheLength - tokenCount || eosCount > MaxVocabSize ||
      tileCount != std::size(GemmaTiles) || temperature != 0.0 ||
      inputBytes != 40 + (tokenCount + eosCount + tileCount) * sizeof(uint64_t))
    throw std::runtime_error("invalid Gemma input parameters");
  const auto *tokenIds = reinterpret_cast<const int64_t *>(header + 5);
  const auto *eosIds = tokenIds + tokenCount;
  const auto *tileIds = reinterpret_cast<const uint64_t *>(eosIds + eosCount);
  for (size_t index = 0; index < tokenCount + eosCount; ++index)
    if (tokenIds[index] < 0 || size_t(tokenIds[index]) >= MaxVocabSize)
      throw std::runtime_error("Gemma token ID exceeds vocabulary");
  for (size_t index = 0; index < tileCount; ++index)
    if (tileIds[index] != GemmaTiles[index])
      throw std::runtime_error(
          "Gemma tile list differs from the compiled plan");
  cpu_set_t cpus;
  CPU_ZERO(&cpus);
  CPU_SET(0, &cpus);
  if (sched_setaffinity(0, sizeof(cpus), &cpus))
    throw std::system_error(errno, std::generic_category(),
                            "pin Gemma main CPU");
  state->request = {std::vector<int64_t>(tokenIds, tokenIds + tokenCount),
                    std::vector<int64_t>(eosIds, eosIds + eosCount), maxTokens,
                    temperature};
  std::vector<int> controllers;
  for (size_t tile : GemmaTiles) {
    bool found = false;
    for (uint32_t hart = 0; hart < BB_HART_NUM; ++hart) {
      auto id = bb_topology_core_id(hart);
      if (id.tile == tile && id.core == 0) {
        controllers.push_back(int(hart));
        found = true;
        break;
      }
    }
    if (!found)
      throw std::runtime_error("Gemma tile has no controller CPU");
  }
  state->execution = createGemmaExecution(directory, controllers);
}

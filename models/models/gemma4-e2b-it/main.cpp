//===- main.cpp ----------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "execution.h"
#include "gemma-plan.h"
#include "generation.h"
#include <cstring>
#include <iostream>
#include <resources.h>
#include <sched.h>
#include <stdexcept>
#include <system_error>

int main(int argc, char **argv) {
  if (argc != 3 && argc != 4)
    throw std::runtime_error("usage: gemma4-e2b-it-run MODEL_DIRECTORY "
                             "INPUT_RESOURCE [RESOURCE_INDEX]");
  const auto directory = std::filesystem::absolute(argv[1]).lexically_normal();
  resources::configure(directory, argc == 4 ? argv[3] : nullptr);
  const auto inputPath = directory / argv[2];
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
  Execution execution(directory,
                      std::vector<size_t>(tileIds, tileIds + tileCount));
  const auto generated =
      generate(execution, std::vector<int64_t>(tokenIds, tokenIds + tokenCount),
               std::vector<int64_t>(eosIds, eosIds + eosCount), maxTokens);
  execution.close();
  std::cout << "{\"token_ids\":[";
  for (size_t index = 0; index < generated.size(); ++index) {
    if (index)
      std::cout << ',';
    std::cout << generated[index];
  }
  std::cout << "]}" << std::endl;
}

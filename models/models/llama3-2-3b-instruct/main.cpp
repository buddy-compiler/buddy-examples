#include "execution.h"
#include "generation.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <time.h>

int main(int argc, char **argv) {
  if (argc != 3 && argc != 4)
    throw std::runtime_error(
        "usage: llama-run MODEL_DIRECTORY INPUT_RESOURCE [RESOURCE_INDEX]");
  const auto directory = std::filesystem::absolute(argv[1]);
  Execution execution(directory, argv[2], argc == 4 ? argv[3] : nullptr);
  const auto &request = execution.request();
  timespec before, after;
  clock_gettime(CLOCK_MONOTONIC, &before);
  const auto generated = generate(execution, request.tokens, request.eos,
                                  request.maxTokens, request.temperature);
  execution.close();
  clock_gettime(CLOCK_MONOTONIC, &after);
  const double elapsed = double(after.tv_sec - before.tv_sec) +
                         double(after.tv_nsec - before.tv_nsec) * 1e-9;
  std::cout << "{\"token_ids\":[";
  for (size_t index = 0; index < generated.size(); ++index) {
    if (index)
      std::cout << ',';
    std::cout << generated[index];
  }
  std::cout << "],\"guest_wall_seconds\":" << elapsed << "}" << std::endl;
}

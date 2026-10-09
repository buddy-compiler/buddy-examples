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
#include "generation.h"
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
  if (argc != 3 && argc != 4)
    throw std::runtime_error("usage: gemma4-e2b-it-run MODEL_DIRECTORY "
                             "INPUT_RESOURCE [RESOURCE_INDEX]");
  const auto directory = std::filesystem::absolute(argv[1]).lexically_normal();
  Execution execution(directory, argv[2], argc == 4 ? argv[3] : nullptr);
  const auto &request = execution.request();
  const auto generated =
      generate(execution, request.tokens, request.eos, request.maxTokens);
  execution.close();
  std::cout << "{\"token_ids\":[";
  for (size_t index = 0; index < generated.size(); ++index) {
    if (index)
      std::cout << ',';
    std::cout << generated[index];
  }
  std::cout << "]}" << std::endl;
}

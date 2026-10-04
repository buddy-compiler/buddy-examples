//===- main.cpp ------------------------------------------------------===//
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

#include "bert-parameters.h"
#include <runtime.h>
#include <buddy/Core/Container.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

extern "C" void _mlir_ciface_forward(MemRef<float, 2> *result,
                                     MemRef<float, 1> *parameters,
                                     MemRef<int8_t, 1> *weights,
                                     MemRef<int64_t, 2> *inputIds,
                                     MemRef<int64_t, 2> *tokenTypes,
                                     MemRef<int64_t, 2> *attentionMask,
                                     MemRef<int64_t, 2> *positions);

template <typename T, size_t Rank>
void readTensor(const std::filesystem::path &path, MemRef<T, Rank> &tensor) {
  std::ifstream input;
  input.exceptions(std::ios::failbit | std::ios::badbit);
  input.open(path, std::ios::binary);
  input.read(reinterpret_cast<char *>(tensor.getData()),
             tensor.getSize() * sizeof(T));
}

int main(int argc, char **argv) {
  const bool single = argc == 4 && std::string(argv[2]) == "--input";
  if (argc != 2 && !single) {
    std::cerr << "usage: bert-run MODEL_DIRECTORY [--input REQUEST_FILE]\n";
    return 1;
  }
  runtime_init(1024 * 1024);
  const std::filesystem::path modelDir = argv[1];
  MemRef<float, 1> parameters({BERT_F32_ELEMENTS});
  MemRef<int8_t, 1> weights({BERT_I8_ELEMENTS});
  readTensor(modelDir / "bert.payload/params.f32", parameters);
  readTensor(modelDir / "bert.payload/weights.bin", weights);

  MemRef<int64_t, 2> inputIds({1, BERT_SEQUENCE_LENGTH});
  MemRef<int64_t, 2> tokenTypes({1, BERT_SEQUENCE_LENGTH});
  MemRef<int64_t, 2> attentionMask({1, BERT_SEQUENCE_LENGTH});
  MemRef<int64_t, 2> positions({1, BERT_SEQUENCE_LENGTH});
  constexpr size_t workspaceBytes = size_t(1) << 27;
  void *workspace = aligned_alloc(64, workspaceBytes);
  if (!workspace)
    throw std::bad_alloc();
  workspace_init(workspace, workspaceBytes);
  std::cout.exceptions(std::ios::failbit | std::ios::badbit);
  std::ifstream requestFile;
  if (single) {
    requestFile.exceptions(std::ios::badbit);
    requestFile.open(argv[3], std::ios::binary);
    if (!requestFile) throw std::runtime_error("cannot open BERT request");
  }
  std::istream &input = single ? static_cast<std::istream &>(requestFile) : std::cin;
  size_t request = 0;
  for (;;) {
    input.read(reinterpret_cast<char *>(inputIds.getData()),
                  BERT_SEQUENCE_LENGTH * sizeof(int64_t));
    if (input.eof() && input.gcount() == 0)
      break;
    if (!input)
      throw std::runtime_error("incomplete BERT input IDs");
    for (auto *tensor : {&tokenTypes, &attentionMask, &positions}) {
      input.read(reinterpret_cast<char *>(tensor->getData()),
                    BERT_SEQUENCE_LENGTH * sizeof(int64_t));
      if (!input)
        throw std::runtime_error("incomplete BERT request");
    }
    workspace_begin(workspace, workspaceBytes);
    MemRef<float, 2> result({1, BERT_NUM_LABELS}, false, 0);
    uint64_t started;
    asm volatile("rdcycle %0" : "=r"(started) :: "memory");
    _mlir_ciface_forward(&result, &parameters, &weights, &inputIds, &tokenTypes,
                         &attentionMask, &positions);
    uint64_t finished;
    asm volatile("rdcycle %0" : "=r"(finished) :: "memory");
    if (single) {
      std::cout << "Cycle count: " << finished-started << "\nLOGITS_F32_BEGIN\n";
      for (unsigned i=0;i<BERT_NUM_LABELS;++i) {
        uint32_t bits;
        std::memcpy(&bits, &result.getData()[i], sizeof(bits));
        std::cout << std::hex << bits << '\n';
      }
      std::cout << std::dec << "LOGITS_F32_END\n";
      for (size_t core=1;core<=core_count();++core)
        std::cout << "Core " << core << " tasks: " << core_submissions(core) << '\n';
    } else {
      std::cout.write(reinterpret_cast<const char *>(result.getData()),
                      BERT_NUM_LABELS * sizeof(float));
    }
    std::cout.flush();
    std::cerr << "request=" << request++
              << " workspace_peak_bytes=" << workspace_peak() << '\n';
    workspace_free(result.release());
  }
  if (single && request != 1) throw std::runtime_error("expected exactly one BERT request");
  free(workspace);
  return 0;
}

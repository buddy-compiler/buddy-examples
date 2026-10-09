#include <runtime.h>
#include <buddy/Core/Container.h>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using Hidden = MemRef<float, 3>;
using Cache = MemRef<float, 4>;
using Floats = MemRef<float, 1>;
using Bytes = MemRef<int8_t, 1>;
using Tokens = MemRef<int64_t, 2>;
using Positions = MemRef<int64_t, 1>;

struct AttentionResult {
  Hidden hidden;
  Cache keys;
  Cache values;
};
struct Parameters {
  const char *directory;
  size_t floats, bytes;
};
struct EmbeddingEntry {
  Parameters parameters;
  void (*run)(Hidden *, Floats *, Tokens *);
};
struct AttentionEntry {
  Parameters parameters;
  void (*run)(AttentionResult *, Floats *, Bytes *, Hidden *, Cache *, Cache *, Positions *);
};
struct FfnEntry {
  Parameters parameters;
  void (*run)(Hidden *, Floats *, Bytes *, Hidden *);
};
using OutputEntry = FfnEntry;

#include "llama-parameters.h"

template <typename T, size_t Rank>
void readTensor(const std::filesystem::path &path, MemRef<T, Rank> &tensor) {
  std::ifstream input;
  input.exceptions(std::ios::failbit | std::ios::badbit);
  input.open(path, std::ios::binary);
  input.read(reinterpret_cast<char *>(tensor.getData()), tensor.getSize() * sizeof(T));
}

struct Weights {
  Floats floats;
  Bytes bytes;
  Weights(const std::filesystem::path &root, Parameters spec)
      : floats({spec.floats}), bytes({spec.bytes}, spec.bytes != 0, 0) {
    readTensor(root / spec.directory / "params.f32", floats);
    if (spec.bytes)
      readTensor(root / spec.directory / "weights.bin", bytes);
  }
};

int main(int argc, char **argv) {
  if (argc != 3)
    throw std::runtime_error("usage: llama-run MODEL_DIRECTORY RANK");
  size_t rank = std::stoul(argv[2]);
  if (rank >= parts)
    throw std::runtime_error("rank exceeds compiled tile count");
  runtime_init(1024 * 1024);
  std::map<std::string, std::unique_ptr<Weights>> weights;
  auto load = [&](Parameters parameters) {
    if (!weights.contains(parameters.directory))
      weights.emplace(parameters.directory, std::make_unique<Weights>(argv[1], parameters));
  };
  if (rank == 0) {
    load(prefill_embedding.parameters);
    load(decode_embedding.parameters);
    load(output.parameters);
  }
  for (size_t layer = 0; layer < layers; ++layer) {
    load(prefill_attention[rank][layer].parameters);
    load(prefill_ffn[rank][layer].parameters);
    load(decode_attention[rank][layer].parameters);
    load(decode_ffn[rank][layer].parameters);
  }
  const std::vector<size_t> cacheShape{1, kvHeads, cacheLength, headSize};
  void *workspace = aligned_alloc(64, workspaceBytes);
  if (!workspace)
    throw std::bad_alloc();
  workspace_init(workspace, workspaceBytes);
  size_t request = 0;
  std::cout.exceptions(std::ios::failbit | std::ios::badbit);
  for (;;) {
    uint64_t header[4];
    std::cin.read(reinterpret_cast<char *>(header), sizeof(header));
    if (std::cin.eof() && std::cin.gcount() == 0)
      break;
    if (!std::cin)
      throw std::runtime_error("incomplete Llama command");
    const auto [operation, count, start, layer] = std::array{header[0], header[1], header[2], header[3]};
    bool prefill = start == 0;
    if (!count || start >= cacheLength || count > cacheLength - start ||
        (prefill ? count > prefillLength : count != 1))
      throw std::runtime_error("invalid Llama token range");
    size_t length = prefill ? prefillLength : 1;
    workspace_begin(workspace, workspaceBytes);
    if (operation == 0) {
      if (rank != 0 || layer != 0)
        throw std::runtime_error("embedding belongs to rank zero");
      Tokens tokens({1, length}, int64_t(0));
      std::cin.read(reinterpret_cast<char *>(tokens.getData()), count * sizeof(int64_t));
      if (!std::cin)
        throw std::runtime_error("incomplete Llama token IDs");
      for (size_t token = 0; token < length; ++token)
        if (tokens.getData()[token] < 0 || size_t(tokens.getData()[token]) >= vocabulary)
          throw std::runtime_error("token ID exceeds vocabulary");
      const auto &embedding = prefill ? prefill_embedding : decode_embedding;
      auto &parameters = *weights.at(embedding.parameters.directory);
      Hidden hidden({1, length, hiddenSize}, false, 0);
      embedding.run(&hidden, &parameters.floats, &tokens);
      std::cout.write(reinterpret_cast<const char *>(hidden.getData()), count * hiddenSize * sizeof(float));
      workspace_free(hidden.release());
    } else if (operation == 1 || operation == 2 || operation == 3) {
      Hidden hidden({1, operation == 1 ? 1 : length, hiddenSize}, 0.0f);
      if (operation == 1 && (rank != 0 || count != 1 || start != 0 || layer != 0))
        throw std::runtime_error("invalid Llama output projection request");
      if (operation != 1 && layer >= layers)
        throw std::runtime_error("invalid Llama layer");
      std::cin.read(reinterpret_cast<char *>(hidden.getData()), count * hiddenSize * sizeof(float));
      if (!std::cin)
        throw std::runtime_error("incomplete Llama hidden states");
      if (operation == 2) {
        Cache keys(cacheShape, 0.0f), values(cacheShape, 0.0f);
        for (Cache *cache : {&keys, &values})
          for (size_t head = 0; head < kvHeads; ++head)
            std::cin.read(reinterpret_cast<char *>(cache->getData() + head * cacheLength * headSize),
                          start * headSize * sizeof(float));
        if (!std::cin)
          throw std::runtime_error("incomplete Llama KV cache");
        Positions positions({length});
        for (size_t token = 0; token < length; ++token)
          positions.getData()[token] = start + token;
        const auto &attention = prefill ? prefill_attention[rank][layer] : decode_attention[rank][layer];
        auto &parameters = *weights.at(attention.parameters.directory);
        AttentionResult result{Hidden({1, length, hiddenSize}, false, 0),
                               Cache(cacheShape, false, 0), Cache(cacheShape, false, 0)};
        attention.run(&result, &parameters.floats, &parameters.bytes, &hidden, &keys, &values, &positions);
        std::cout.write(reinterpret_cast<const char *>(result.hidden.getData()), count * hiddenSize * sizeof(float));
        for (Cache *cache : {&result.keys, &result.values})
          for (size_t head = 0; head < kvHeads; ++head)
            std::cout.write(reinterpret_cast<const char *>(cache->getData() +
                                                          (head * cacheLength + start) * headSize),
                            count * headSize * sizeof(float));
        workspace_free(result.hidden.release());
        workspace_free(result.keys.release());
        workspace_free(result.values.release());
      } else {
        const auto &stage = operation == 1 ? output :
                            (prefill ? prefill_ffn[rank][layer] : decode_ffn[rank][layer]);
        auto &parameters = *weights.at(stage.parameters.directory);
        size_t width = operation == 1 ? vocabulary : hiddenSize;
        Hidden result({1, operation == 1 ? 1 : length, width}, false, 0);
        stage.run(&result, &parameters.floats, &parameters.bytes, &hidden);
        std::cout.write(reinterpret_cast<const char *>(result.getData()), count * width * sizeof(float));
        workspace_free(result.release());
      }
    } else {
      throw std::runtime_error("unknown Llama command");
    }
    std::cout.flush();
    std::cerr << "request=" << request++ << " rank=" << rank << " operation=" << operation
              << " layer=" << layer << " workspace_peak_bytes=" << workspace_peak() << '\n';
  }
  free(workspace);
}

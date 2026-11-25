// Qwen3-0.6B compiled for MAGIA, run on this machine with the mesh's
// products in software (the MAGIA hart's reference magia_gemm): the prompt,
// then one token a step through the KV cache, against the tokens the model
// writes in PyTorch with the same products (buddy-pytorch-importer.py).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

constexpr size_t layers = 28;
constexpr size_t heads = 8, capacity = 256, width = 128;
constexpr size_t vocabulary = 151936;

// A memref of the compiled graph.
template <typename T, int N> struct Buffer {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};

// What a step returns: the logits of its tokens and the updated cache, every
// buffer owned by the caller.
struct Results {
  Buffer<float, 3> logits;
  Buffer<float, 4> keys[layers];
  Buffer<float, 4> values[layers];
};

extern "C" void _mlir_ciface_forward();

// The entry takes the FP16 weights and the FP32 parameters, then the tokens,
// the first position and the cache of every layer in the order of
// input-order.txt: 2 + 2 * layers buffers.
template <size_t... I>
void forward(Results *results, Buffer<uint16_t, 1> *weights,
             Buffer<float, 1> *parameters, void **inputs,
             std::index_sequence<I...>) {
  using Entry = void (*)(Results *, Buffer<uint16_t, 1> *, Buffer<float, 1> *,
                         decltype((void)I, (void *)nullptr)...);
  reinterpret_cast<Entry>(&_mlir_ciface_forward)(results, weights, parameters,
                                                 inputs[I]...);
}

template <typename T> std::vector<T> load(const std::string &path) {
  std::ifstream file;
  file.exceptions(std::ios::failbit | std::ios::badbit);
  file.open(path, std::ios::binary | std::ios::ate);
  std::vector<T> values(size_t(file.tellg()) / sizeof(T));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(values.data()), values.size() * sizeof(T));
  return values;
}

int main(int, char **argv) {
  const std::string directory = argv[1];
  auto halves = load<uint16_t>(directory + "/arg0.data");
  auto floats = load<float>(directory + "/arg1.data");
  const auto prompt = load<long long>(directory + "/check-prompt.data");
  const auto expected = load<long long>(directory + "/check-tokens.data");
  const auto logits = load<float>(directory + "/check-logits.data");
  std::ifstream orderFile(directory + "/input-order.txt");
  std::vector<size_t> order;
  for (size_t index; orderFile >> index;)
    order.push_back(index);
  if (order.size() != 2 + 2 * layers)
    throw std::runtime_error("Invalid graph input order");
  Buffer<uint16_t, 1> weights{
      halves.data(), halves.data(), 0, {intptr_t(halves.size())}, {1}};
  Buffer<float, 1> parameters{
      floats.data(), floats.data(), 0, {intptr_t(floats.size())}, {1}};

  // The cache: the keys of every layer, then the values, empty.
  Buffer<float, 4> cache[2 * layers];
  for (auto &buffer : cache) {
    auto *data =
        static_cast<float *>(calloc(heads * capacity * width, sizeof(float)));
    buffer = {data,
              data,
              0,
              {1, heads, capacity, width},
              {heads * capacity * width, capacity * width, width, 1}};
  }

  std::vector<long long> tokens = prompt, written;
  float largest = 0, scale = 0;
  long long start = 0;
  for (size_t step = 0; step < expected.size(); ++step) {
    Buffer<long long, 2> ids{tokens.data(),
                             tokens.data(),
                             0,
                             {1, intptr_t(tokens.size())},
                             {intptr_t(tokens.size()), 1}};
    Buffer<long long, 1> first{&start, &start, 0, {1}, {1}};
    void *example[2 + 2 * layers] = {&ids, &first};
    for (size_t index = 0; index < 2 * layers; ++index)
      example[2 + index] = &cache[index];
    void *inputs[2 + 2 * layers];
    for (size_t index = 0; index < order.size(); ++index)
      inputs[index] = example[order[index]];
    Results results;
    forward(&results, &weights, &parameters, inputs,
            std::make_index_sequence<2 + 2 * layers>{});
    // The next token: the largest logit of the last position.
    const float *last = results.logits.aligned + results.logits.offset +
                        (tokens.size() - 1) * vocabulary;
    const long long token = std::max_element(last, last + vocabulary) - last;
    for (size_t i = 0; i < vocabulary; ++i) {
      largest =
          std::max(largest, std::fabs(last[i] - logits[step * vocabulary + i]));
      scale = std::max(scale, std::fabs(logits[step * vocabulary + i]));
    }
    free(results.logits.allocated);
    for (size_t layer = 0; layer < layers; ++layer) {
      free(cache[layer].allocated);
      free(cache[layers + layer].allocated);
      cache[layer] = results.keys[layer];
      cache[layers + layer] = results.values[layer];
    }
    start += tokens.size();
    tokens = {token};
    written.push_back(token);
    std::cout << "token " << step << ": " << token << " (the reference "
              << expected[step] << ")" << std::endl;
  }
  size_t same = 0;
  for (size_t step = 0; step < expected.size(); ++step)
    same += written[step] == expected[step];
  std::cout << "qwen3-0-6b on magia: " << same << '/' << expected.size()
            << " tokens as the reference, largest logit difference " << largest
            << " (relative " << largest / scale << ")\n";
  return same == expected.size() ? 0 : 1;
}

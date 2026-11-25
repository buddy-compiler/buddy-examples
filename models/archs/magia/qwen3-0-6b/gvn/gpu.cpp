// The GPU side of GVN for MAGIA Qwen3-0.6B: runs the linalg graph the MAGIA
// compiler starts from on the recipe's FP16 weights and FP32 parameters for
// the prompt step of its check (check-prompt.data, from position 0, the cache
// empty). The entry returns, instead of the logits and the cache, the values
// the slice checks name (gvn-expose, gvn-forward-results); they are written
// one after another, and their sizes, one line each, to shapes.txt beside the
// values file (their tokens are dynamic).
//
// gpu <model output dir> <values file> <rank:element bytes:bytes>...
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

constexpr size_t layers = 28;
constexpr size_t heads = 8, capacity = 256, width = 128;

// A memref of the compiled graph.
template <typename T, int N> struct Buffer {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};

extern "C" void _mlir_ciface_forward();

// The entry takes the FP16 weights and the FP32 parameters, then the tokens,
// the first position and the cache of every layer in the order of
// input-order.txt: 2 + 2 * layers buffers.
template <size_t... I>
void forward(void *results, Buffer<uint16_t, 1> *weights,
             Buffer<float, 1> *parameters, void **inputs,
             std::index_sequence<I...>) {
  using Entry = void (*)(void *, Buffer<uint16_t, 1> *, Buffer<float, 1> *,
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

int main(int argc, char **argv) {
  const std::string directory = argv[1];
  auto halves = load<uint16_t>(directory + "/arg0.data");
  auto floats = load<float>(directory + "/arg1.data");
  auto tokens = load<long long>(directory + "/check-prompt.data");
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
  long long start = 0;
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

  // The results: one descriptor per value, (allocated, aligned, offset,
  // sizes, strides) of its rank.
  struct Value {
    int64_t rank, element, bytes;
  };
  std::vector<Value> values;
  size_t descriptorBytes = 0;
  for (int i = 3; i < argc; ++i) {
    Value value;
    std::sscanf(argv[i], "%ld:%ld:%ld", &value.rank, &value.element,
                &value.bytes);
    values.push_back(value);
    descriptorBytes += 8 * (3 + 2 * value.rank);
  }
  std::vector<int64_t> results(descriptorBytes / 8);
  forward(results.data(), &weights, &parameters, inputs,
          std::make_index_sequence<2 + 2 * layers>{});

  std::ofstream out(argv[2], std::ios::binary);
  std::ofstream shapes(
      std::string(argv[2]).substr(0, std::string(argv[2]).rfind('/') + 1) +
      "shapes.txt");
  const int64_t *descriptor = results.data();
  for (const Value &value : values) {
    const int64_t *sizes = descriptor + 3, *strides = sizes + value.rank;
    int64_t elements = 1;
    for (int64_t dim = value.rank - 1; dim >= 0; --dim) {
      if (strides[dim] != elements && sizes[dim] != 1)
        throw std::runtime_error("a value is not contiguous");
      elements *= sizes[dim];
    }
    for (int64_t dim = 0; dim < value.rank; ++dim)
      shapes << (dim ? " " : "") << sizes[dim];
    shapes << "\n";
    const char *aligned = reinterpret_cast<const char *>(descriptor[1]);
    out.write(aligned + descriptor[2] * value.element,
              elements * value.element);
    descriptor += 3 + 2 * value.rank;
  }
  return 0;
}

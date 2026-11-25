// The GPU side of GVN for gemmini ResNet18: runs the linalg graph the Gemmini
// compiler starts from on the recipe's parameters and image. The entry
// returns, instead of the logits, the values the slice checks name
// (gvn-expose, gvn-forward-results); they are written one after another.
//
// gpu <model output dir> <values file> <rank:element bytes:bytes>...
#include "MemRef.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using model::MemRef;

extern "C" void _mlir_ciface_forward(void *results, MemRef<float, 1> *,
                                     MemRef<int32_t, 1> *, MemRef<int8_t, 1> *,
                                     MemRef<float, 4> *);

static std::vector<char> load(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::fprintf(stderr, "cannot read %s\n", path.c_str());
    std::exit(1);
  }
  std::vector<char> bytes(file.tellg());
  file.seekg(0);
  file.read(bytes.data(), bytes.size());
  return bytes;
}

template <typename T, unsigned Rank>
static MemRef<T, Rank> view(std::vector<char> &bytes,
                            const int64_t (&sizes)[Rank]) {
  MemRef<T, Rank> memref{};
  memref.allocated = memref.aligned = reinterpret_cast<T *>(bytes.data());
  int64_t stride = 1;
  for (int dim = Rank - 1; dim >= 0; --dim) {
    memref.sizes[dim] = sizes[dim];
    memref.strides[dim] = stride;
    stride *= sizes[dim];
  }
  if (stride * sizeof(T) != bytes.size()) {
    std::fprintf(stderr, "unexpected size of an input\n");
    std::exit(1);
  }
  return memref;
}

int main(int argc, char **argv) {
  const std::string directory = argv[1];
  std::vector<char> parameters = load(directory + "/arg0.data");
  std::vector<char> biases = load(directory + "/arg1.data");
  std::vector<char> weights = load(directory + "/arg2.data");
  std::vector<char> pixels = load(directory + "/inputs.data");
  auto f32 = view<float, 1>(parameters, {int64_t(parameters.size() / 4)});
  auto i32 = view<int32_t, 1>(biases, {int64_t(biases.size() / 4)});
  auto i8 = view<int8_t, 1>(weights, {int64_t(weights.size())});
  auto input = view<float, 4>(pixels, {1, 3, 224, 224});

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
  _mlir_ciface_forward(results.data(), &f32, &i32, &i8, &input);

  std::ofstream out(argv[2], std::ios::binary);
  const int64_t *descriptor = results.data();
  for (const Value &value : values) {
    const char *aligned = reinterpret_cast<const char *>(descriptor[1]);
    out.write(aligned + descriptor[2] * value.element, value.bytes);
    descriptor += 3 + 2 * value.rank;
  }
  return 0;
}

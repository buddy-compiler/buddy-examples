// The GPU side of GVN for MAGIA BERT (emotion): runs the linalg graph the
// MAGIA compiler starts from on the recipe's FP16 weights, FP32 parameters and
// its first validation sentence (check-*.data). The entry returns, instead of
// the logits, the values the slice checks name (gvn-expose,
// gvn-forward-results); they are written one after another.
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

extern "C" void _mlir_ciface_forward(void *results, MemRef<uint16_t, 1> *,
                                     MemRef<float, 1> *, MemRef<int64_t, 2> *,
                                     MemRef<int64_t, 2> *, MemRef<int64_t, 2> *,
                                     MemRef<int64_t, 2> *);

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
  std::vector<char> weights = load(directory + "/arg0.data");
  std::vector<char> parameters = load(directory + "/arg1.data");
  // The first sentence of each: 32 tokens.
  std::vector<char> ids = load(directory + "/check-ids.data");
  std::vector<char> positions = load(directory + "/check-positions.data");
  std::vector<char> types = load(directory + "/check-types.data");
  std::vector<char> mask = load(directory + "/check-mask.data");
  for (std::vector<char> *tokens : {&ids, &positions, &types, &mask})
    tokens->resize(32 * sizeof(int64_t));
  auto f16 = view<uint16_t, 1>(weights, {int64_t(weights.size() / 2)});
  auto f32 = view<float, 1>(parameters, {int64_t(parameters.size() / 4)});
  auto id = view<int64_t, 2>(ids, {1, 32});
  auto position = view<int64_t, 2>(positions, {1, 32});
  auto type = view<int64_t, 2>(types, {1, 32});
  auto masked = view<int64_t, 2>(mask, {1, 32});

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
  _mlir_ciface_forward(results.data(), &f16, &f32, &id, &position, &type,
                       &masked);

  std::ofstream out(argv[2], std::ios::binary);
  const int64_t *descriptor = results.data();
  for (const Value &value : values) {
    const char *aligned = reinterpret_cast<const char *>(descriptor[1]);
    out.write(aligned + descriptor[2] * value.element, value.bytes);
    descriptor += 3 + 2 * value.rank;
  }
  return 0;
}

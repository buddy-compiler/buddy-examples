// BERT (emotion) compiled for MAGIA, run on this machine with the mesh's
// products in software (the MAGIA hart's reference magia_gemm): the first
// validation sentences against the model in PyTorch with the same products
// (buddy-pytorch-importer.py).
#include <buddy/Core/Container.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// The FP16 weights and the FP32 parameters; the imported ABI orders inputs as
// IDs, positions, types, then mask.
extern "C" void
_mlir_ciface_forward(MemRef<float, 2> *output, MemRef<uint16_t, 1> *weights,
                     MemRef<float, 1> *parameters, MemRef<long long, 2> *ids,
                     MemRef<long long, 2> *positions,
                     MemRef<long long, 2> *types, MemRef<long long, 2> *mask);

template <typename T> std::vector<T> load(const std::string &path) {
  std::ifstream file;
  file.exceptions(std::ios::failbit | std::ios::badbit);
  file.open(path, std::ios::binary | std::ios::ate);
  std::vector<T> values(size_t(file.tellg()) / sizeof(T));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(values.data()), values.size() * sizeof(T));
  return values;
}

MemRef<long long, 2> sentence(const std::vector<long long> &values,
                              size_t sample) {
  intptr_t sizes[2] = {1, 32};
  return MemRef<long long, 2>(values.data() + sample * 32, sizes);
}

int main(int, char **argv) {
  const std::string directory = argv[1];
  const auto halves = load<uint16_t>(directory + "/arg0.data");
  const auto values = load<float>(directory + "/arg1.data");
  const auto allIds = load<long long>(directory + "/check-ids.data");
  const auto allTypes = load<long long>(directory + "/check-types.data");
  const auto allMasks = load<long long>(directory + "/check-mask.data");
  const auto allPositions =
      load<long long>(directory + "/check-positions.data");
  const auto logits = load<float>(directory + "/check-logits.data");
  intptr_t halfCount[1] = {intptr_t(halves.size())};
  MemRef<uint16_t, 1> weights(halves.data(), halfCount);
  intptr_t count[1] = {intptr_t(values.size())};
  MemRef<float, 1> parameters(values.data(), count);

  const size_t sentences = allIds.size() / 32;
  size_t same = 0;
  float largest = 0;
  for (size_t sample = 0; sample < sentences; ++sample) {
    auto ids = sentence(allIds, sample);
    auto types = sentence(allTypes, sample);
    auto mask = sentence(allMasks, sample);
    auto positions = sentence(allPositions, sample);
    // Each invocation returns an owned output descriptor.
    MemRef<float, 2> output(std::vector<size_t>{1, 6}, false, 0);
    _mlir_ciface_forward(&output, &weights, &parameters, &ids, &positions,
                         &types, &mask);
    const float *got = output.getData();
    const float *expected = logits.data() + sample * 6;
    same += std::max_element(got, got + 6) - got ==
            std::max_element(expected, expected + 6) - expected;
    for (size_t i = 0; i < 6; ++i)
      largest = std::max(largest, std::fabs(got[i] - expected[i]));
  }
  std::cout << "bert-base-uncased-emotion on magia: " << same << '/'
            << sentences
            << " predictions as the reference, largest logit difference "
            << largest << '\n';
  return same == sentences ? 0 : 1;
}

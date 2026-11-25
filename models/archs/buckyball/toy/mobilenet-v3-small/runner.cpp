// MobileNetV3-Small on the toy hart under Linux: the prepared image of the
// recipe against its reference, as baremetal.cpp checks it.
#include "SharedLibrary.h"
#include "model.h"
#include "testutils.h"
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using mobilenet::Forward;
using model::openLibrary;
using model::SharedLibrary;

template <typename T> std::vector<T> load(const std::string &path) {
  std::ifstream file;
  file.exceptions(std::ios::failbit | std::ios::badbit);
  file.open(path, std::ios::binary | std::ios::ate);
  const size_t bytes = file.tellg();
  if (bytes % sizeof(T))
    throw std::runtime_error("Invalid parameter byte count: " + path);
  std::vector<T> data(bytes / sizeof(T));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(data.data()), bytes);
  return data;
}
class MobileNetRunner final : public buddy::runtime::InferenceRunner {
public:
  void run(const buddy::runtime::RunConfig &config) override {
    const auto manifest =
        buddy::runtime::ModelManifest::loadFromRax(config.raxPath);
    std::vector<SharedLibrary> runtimes;
    for (const auto &path : manifest.dependentSoPaths)
      runtimes.push_back(openLibrary(path));
    auto library = openLibrary(manifest.soPath);
    const auto &entry = manifest.codeObjects.at(0).entrySymbol;
    auto forward =
        reinterpret_cast<Forward>(dlsym(library.get(), entry.c_str()));
    if (!forward)
      throw std::runtime_error("Missing " + entry + ": " +
                               std::string(dlerror()));

    auto parameterData = load<float>(manifest.weightPaths.at(0));
    auto biases = load<int32_t>(manifest.weightPaths.at(1));
    auto weights = load<int8_t>(manifest.weightPaths.at(2));
    mobilenet::Bias i32{biases.data(),
                        biases.data(),
                        0,
                        {static_cast<int64_t>(biases.size())},
                        {1}};
    mobilenet::Weights i8{weights.data(),
                          weights.data(),
                          0,
                          {static_cast<int64_t>(weights.size())},
                          {1}};
    mobilenet::Parameters parameters{
        parameterData.data(),
        parameterData.data(),
        0,
        {static_cast<int64_t>(parameterData.size())},
        {1}};

    std::vector<float> inputData(3 * 224 * 224), expected(1000);
    std::ifstream file;
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.open(config.inputPath, std::ios::binary);
    file.read(reinterpret_cast<char *>(inputData.data()),
              inputData.size() * sizeof(float));
    file.close();
    file.open(config.referencePath, std::ios::binary);
    file.read(reinterpret_cast<char *>(expected.data()),
              expected.size() * sizeof(float));
    mobilenet::Input input{inputData.data(),
                           inputData.data(),
                           0,
                           {1, 3, 224, 224},
                           {150528, 50176, 224, 1}};
    mobilenet::Output output{};
    const uint64_t start = read_counter();
    forward(&output, &parameters, &i32, &i8, &input);
    const uint64_t elapsed = read_counter() - start;
    const float *values = output.data();
    unsigned best = 0, gold = 0;
    for (unsigned i = 1; i < 1000; ++i) {
      if (values[i] > values[best])
        best = i;
      if (expected[i] > expected[gold])
        gold = i;
    }
    printf("Classification: %u, reference %u\nInference cycles: %lu\n", best,
           gold, static_cast<unsigned long>(elapsed));
    // An INT8 code on a rounding boundary flips with the order of the FP32 sums
    // before it (the software depthwise filters and gates against PyTorch):
    // FP32 noise of 2e-7 there moves the reference's logits by up to 0.040 of
    // the largest one, the class unchanged. The bound is 0.05 and the class
    // must match.
    const int failed =
        check_output(values, expected.data(), 1000, 5e-2f) || best != gold;
    std::free(output.allocated);
    if (failed)
      throw std::runtime_error("Output differs from the reference");
  }
};

} // namespace

extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new MobileNetRunner();
}

extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

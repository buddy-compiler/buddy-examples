#include "checked.h"
#include "resources.h"
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>
#include <fstream>
#include <vector>

// The mesh reads memory that is the DDR at a known address in place
// (the hart's stack/runtime/mesh/linux.c).
extern "C" void magia_region(const void *address, uint64_t physical,
                             uint64_t bytes);

namespace {
class Runner final : public buddy::runtime::InferenceRunner {
public:
  void run(const buddy::runtime::RunConfig &config) override {
    const auto manifest =
        buddy::runtime::ModelManifest::loadFromRax(config.raxPath);
    resources::configure(
        std::stoull(manifest.moduleAttrs.at("image_base"), nullptr, 0));
    const auto &paths = manifest.weightPaths;
    const auto weightBytes = resources::image.files.at(paths.at(0)).bytes;
    const auto parameterBytes = resources::image.files.at(paths.at(1)).bytes;
    resources::Mapping weights(paths.at(0), weightBytes, 64);
    resources::Mapping parameters(paths.at(1), parameterBytes, 64);
    // The weights of the Linears stay in the model region for the mesh.
    magia_region(weights.data(),
                 resources::image.base +
                     resources::image.files.at(paths.at(0)).offset,
                 weightBytes);
    std::ifstream file;
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.open(config.inputPath, std::ios::binary | std::ios::ate);
    const size_t bytes = file.tellg();
    if (bytes % sizeof(int64_t))
      throw std::runtime_error("Invalid token data size");
    std::vector<int64_t> input(bytes / sizeof(int64_t));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(input.data()), bytes);
    if (check_model(static_cast<const uint16_t *>(weights.data()),
                    weightBytes / sizeof(uint16_t),
                    static_cast<const float *>(parameters.data()),
                    parameterBytes / sizeof(float), input.data(), input.size()))
      throw std::runtime_error("MAGIA inference differs from its reference");
  }
};
} // namespace

extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new Runner();
}
extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

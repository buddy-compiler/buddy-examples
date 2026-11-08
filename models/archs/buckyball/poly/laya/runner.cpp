#include "SharedLibrary.h"
#include "model.h"
#include "testutils.h"
#include <algorithm>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>
#include <fstream>
#include <iostream>
#include <resources.h>

namespace {
class LayaRunner final : public buddy::runtime::InferenceRunner {
public:
  void run(const buddy::runtime::RunConfig &config) override {
    const auto manifest =
        buddy::runtime::ModelManifest::loadFromRax(config.raxPath);
    auto library = model::openLibrary(manifest.soPath);
    auto forward = reinterpret_cast<int (*)(void **)>(
        dlsym(library.get(), manifest.codeObjects.at(0).entrySymbol.c_str()));
    if (!forward)
      throw std::runtime_error("Missing laya_run");
    resources::configure(
        std::stoull(manifest.moduleAttrs.at("image_base"), nullptr, 0));
    const auto &p = manifest.weightPaths.at(0), &w = manifest.weightPaths.at(1);
    resources::Mapping parameters(p, resources::image.files.at(p).bytes, 64);
    resources::Mapping weights(w, resources::image.files.at(w).bytes, 64);
    Input input;
    std::ifstream source;
    source.exceptions(std::ios::failbit | std::ios::badbit);
    source.open(config.inputPath, std::ios::binary);
    source.read(reinterpret_cast<char *>(&input), sizeof(input));
    float choices[16], actions[2];
    void *arguments[] = {parameters.data(), weights.data(), input.tokens,
                         (void *)128,       input.mask,     (void *)128,
                         input.positions,   input.valid,    input.type,
                         choices,           actions};
    const auto begin = read_time_ns();
    if (forward(arguments))
      throw std::runtime_error("Device program failed");
    const double elapsed = (read_time_ns() - begin) * 1e-9;
    std::cout << "Choice: " << std::max_element(choices, choices + 16) - choices
              << "\nAction: " << (actions[1] > actions[0])
              << "\nInference time: " << elapsed << "s\n";
    if (!config.outputPath.empty()) {
      std::ofstream file;
      file.exceptions(std::ios::failbit | std::ios::badbit);
      file.open(config.outputPath, std::ios::binary);
      file.write(reinterpret_cast<const char *>(choices), sizeof(choices));
      file.write(reinterpret_cast<const char *>(actions), sizeof(actions));
    }
  }
};
} // namespace
extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new LayaRunner();
}
extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

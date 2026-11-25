#include "SharedLibrary.h"
#include "model.h"
#include "testutils.h"
#include <algorithm>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <resources.h>
#include <vector>

namespace {
class Pi0Runner final : public buddy::runtime::InferenceRunner {
public:
  void run(const buddy::runtime::RunConfig &config) override {
    const auto manifest =
        buddy::runtime::ModelManifest::loadFromRax(config.raxPath);
    auto library = model::openLibrary(manifest.soPath);
    auto forward = reinterpret_cast<int (*)(void **)>(
        dlsym(library.get(), manifest.codeObjects.at(0).entrySymbol.c_str()));
    if (!forward)
      throw std::runtime_error("Missing pi0_run");
    resources::configure(
        std::stoull(manifest.moduleAttrs.at("image_base"), nullptr, 0));
    std::vector<std::unique_ptr<resources::Mapping>> buffers;
    std::vector<void *> arguments;
    for (const auto &constant : manifest.constants) {
      buffers.push_back(std::make_unique<resources::Mapping>(
          constant.path, resources::image.files.at(constant.path).bytes, 64));
      arguments.push_back(buffers.back()->data());
    }
    auto input = std::make_unique<Observation>();
    std::ifstream source;
    source.exceptions(std::ios::failbit | std::ios::badbit);
    source.open(config.inputPath, std::ios::binary);
    source.read(reinterpret_cast<char *>(input.get()), sizeof(Observation));
    arguments.insert(arguments.end(),
                     {input->images, input->tokens, input->masks, input->state,
                      input->noise});
    std::vector<float> actions(
        std::stoull(manifest.moduleAttrs.at("action_elements")));
    arguments.push_back(actions.data());
    const auto begin = read_time_ns();
    const uint64_t first = read_counter();
    if (forward(arguments.data()))
      throw std::runtime_error("Device program failed");
    const uint64_t cycles = read_counter() - first;
    const double elapsed = (read_time_ns() - begin) * 1e-9;
    printf("Inference cycles: %lu\n", static_cast<unsigned long>(cycles));
    // The actions against PyTorch (actions.data), as baremetal.cpp reports
    // them: bit by bit and at most.
    if (!config.referencePath.empty()) {
      std::vector<float> expected(actions.size());
      std::ifstream golden;
      golden.exceptions(std::ios::failbit | std::ios::badbit);
      golden.open(config.referencePath, std::ios::binary);
      golden.read(reinterpret_cast<char *>(expected.data()),
                  expected.size() * sizeof(float));
      unsigned different = 0;
      float largest = 0;
      for (size_t i = 0; i < actions.size(); ++i) {
        different += memcmp(&actions[i], &expected[i], sizeof(float)) != 0;
        largest = std::max(largest, std::abs(actions[i] - expected[i]));
      }
      printf("actions: %u of %zu differ from PyTorch, max |difference| %.9f\n",
             different, actions.size(), largest);
    }
    std::cout << "Actions: " << actions.size()
              << "\nInference time: " << elapsed << "s\n";
    if (!config.outputPath.empty()) {
      std::ofstream file;
      file.exceptions(std::ios::failbit | std::ios::badbit);
      file.open(config.outputPath, std::ios::binary);
      file.write(reinterpret_cast<const char *>(actions.data()),
                 actions.size() * sizeof(float));
    }
  }
};
} // namespace
extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new Pi0Runner();
}
extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

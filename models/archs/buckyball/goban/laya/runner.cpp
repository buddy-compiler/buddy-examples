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
    auto forward = reinterpret_cast<laya::Forward>(
        dlsym(library.get(), manifest.codeObjects.at(0).entrySymbol.c_str()));
    if (!forward)
      throw std::runtime_error("Missing laya_run");
    resources::configure(
        std::stoull(manifest.moduleAttrs.at("image_base"), nullptr, 0));
    const auto &pfile = manifest.weightPaths.at(0);
    const auto &bfile = manifest.weightPaths.at(1);
    const auto &wfile = manifest.weightPaths.at(2);
    const auto pb = resources::image.files.at(pfile).bytes;
    const auto bb = resources::image.files.at(bfile).bytes;
    const auto wb = resources::image.files.at(wfile).bytes;
    resources::Mapping pm(pfile, pb, 64), bm(bfile, bb, 64), wm(wfile, wb, 64);
    const auto *p = static_cast<const float *>(pm.data());
    const auto *b = static_cast<const int32_t *>(bm.data());
    const auto *w = static_cast<const int8_t *>(wm.data());
    laya::Parameters parameters{p, p, 0, {int64_t(pb / 4)}, {1}};
    laya::Bias bias{b, b, 0, {int64_t(bb / 4)}, {1}};
    laya::Weights weights{w, w, 0, {int64_t(wb)}, {1}};
    laya::Input input;
    std::ifstream file;
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.open(config.inputPath, std::ios::binary);
    file.read(reinterpret_cast<char *>(&input), sizeof(input));
    laya::Tokens tokens{input.tokens, input.tokens, 0, {1, 128}, {128, 1}};
    laya::Tokens mask{input.mask, input.mask, 0, {1, 128}, {128, 1}};
    laya::Tokens positions{
        input.positions, input.positions, 0, {1, 16}, {16, 1}};
    laya::Valid valid{input.valid, input.valid, 0, {1, 16}, {16, 1}};
    laya::Type type{input.type, input.type, 0, {1}, {1}};
    laya::Output output{};
    const auto begin = read_time_ns();
    const uint64_t first = read_counter();
    forward(&output, &parameters, &bias, &weights, &tokens, &mask, &positions,
            &valid, &type);
    const uint64_t cycles = read_counter() - first;
    const double elapsed = (read_time_ns() - begin) * 1e-9;
    // Against the reference of the recipe (expected.data), as baremetal.cpp
    // checks it: the same lines and bound, and a failure when they differ.
    if (!config.referencePath.empty()) {
      float expected[18];
      std::ifstream golden;
      golden.exceptions(std::ios::failbit | std::ios::badbit);
      golden.open(config.referencePath, std::ios::binary);
      golden.read(reinterpret_cast<char *>(expected), sizeof(expected));
      float scores[18];
      std::copy_n(output.choice.data(), 16, scores);
      std::copy_n(output.action.data(), 2, scores + 16);
      unsigned choice = 0, gold_choice = 0;
      for (unsigned i = 1; i < 16; ++i) {
        if (scores[i] > scores[choice])
          choice = i;
        if (expected[i] > expected[gold_choice])
          gold_choice = i;
      }
      const unsigned action = scores[17] > scores[16];
      const unsigned gold_action = expected[17] > expected[16];
      printf("Choice: %u, reference %u\nAction: %u, reference %u\n"
             "Inference cycles: %lu\n",
             choice, gold_choice, action, gold_action,
             static_cast<unsigned long>(cycles));
      const int failed = check_output(scores, expected, 18, 2e-2f) ||
                         choice != gold_choice || action != gold_action;
      free(output.choice.allocated);
      free(output.action.allocated);
      if (failed)
        throw std::runtime_error("Output differs from the reference");
      return;
    }
    const auto choice =
        std::max_element(output.choice.data(), output.choice.data() + 16) -
        output.choice.data();
    const auto action =
        std::max_element(output.action.data(),
                         output.action.data() + output.action.sizes[1]) -
        output.action.data();
    std::cout << "Choice: " << choice << "\nAction: " << action
              << "\nInference time: " << elapsed << "s\n";
    if (!config.outputPath.empty()) {
      std::ofstream dump;
      dump.exceptions(std::ios::failbit | std::ios::badbit);
      dump.open(config.outputPath, std::ios::binary);
      dump.write(reinterpret_cast<const char *>(output.choice.data()),
                 16 * sizeof(float));
      dump.write(reinterpret_cast<const char *>(output.action.data()),
                 output.action.sizes[1] * sizeof(float));
    }
    free(output.choice.allocated);
    free(output.action.allocated);
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

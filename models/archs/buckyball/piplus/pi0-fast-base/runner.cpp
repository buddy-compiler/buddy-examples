#include "SharedLibrary.h"
#include "model.h"
#include "testutils.h"
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <resources.h>
#include <vector>

namespace {
constexpr unsigned keys = PREFIX_KEYS + GENERATED + OWN_KEYS;

// The prefill once, then a decoding step per token until the EOS; between
// the steps, the keys and values of the generated tokens and the bias that
// sees them (baremetal.cpp).
class Pi0FastRunner final : public buddy::runtime::InferenceRunner {
public:
  void run(const buddy::runtime::RunConfig &config) override {
    const auto manifest =
        buddy::runtime::ModelManifest::loadFromRax(config.raxPath);
    auto library = model::openLibrary(manifest.soPath);
    auto prefill = reinterpret_cast<int (*)(void **)>(
        dlsym(library.get(), manifest.codeObjects.at(0).entrySymbol.c_str()));
    auto decode = reinterpret_cast<int (*)(void **)>(
        dlsym(library.get(), "pi0_fast_decode"));
    if (!prefill || !decode)
      throw std::runtime_error("Missing pi0_fast_prefill or pi0_fast_decode");
    resources::configure(
        std::stoull(manifest.moduleAttrs.at("image_base"), nullptr, 0));
    std::vector<std::unique_ptr<resources::Mapping>> buffers;
    std::vector<void *> packs;
    for (const auto &constant : manifest.constants) {
      buffers.push_back(std::make_unique<resources::Mapping>(
          constant.path, resources::image.files.at(constant.path).bytes, 64));
      packs.push_back(buffers.back()->data());
    }
    auto input = std::make_unique<Observation>();
    std::ifstream source;
    source.exceptions(std::ios::failbit | std::ios::badbit);
    source.open(config.inputPath, std::ios::binary);
    source.read(reinterpret_cast<char *>(input.get()), sizeof(Observation));

    std::vector<float> logits(VOCABULARY), bias(keys);
    std::vector<std::vector<float>> prefix(
        2 * LAYERS, std::vector<float>(PREFIX_KEYS * HEAD_DIM)),
        before(2 * LAYERS, std::vector<float>(GENERATED * HEAD_DIM)),
        own(2 * LAYERS, std::vector<float>(HEAD_DIM));
    int64_t position = 0, token = 0, place = 0;
    // The rows of the embedding table of the tokens, which no graph holds.
    resources::Mapping table(
        "ddr/embedding.data",
        resources::image.files.at("ddr/embedding.data").bytes, 64);
    const auto *rows = static_cast<const float *>(table.data());
    std::vector<float> prompt(201 * HIDDEN), row(HIDDEN);
    for (unsigned i = 0; i < 201; ++i)
      memcpy(prompt.data() + i * HIDDEN, rows + input->tokens[i] * HIDDEN,
             HIDDEN * sizeof(float));
    std::vector<void *> arguments(packs.begin(), packs.begin() + 4);
    arguments.insert(arguments.end(),
                     {input->images, prompt.data(), input->masks, logits.data(),
                      &position, bias.data()});
    for (auto &buffer : prefix)
      arguments.push_back(buffer.data());
    const auto begin = read_time_ns();
    const uint64_t first = read_counter();
    if (prefill(arguments.data()))
      throw std::runtime_error("Prefill program failed");
    auto likeliest = [&] {
      int64_t best = 0;
      for (int64_t i = 1; i < VOCABULARY; ++i)
        if (logits[i] > logits[best])
          best = i;
      return best;
    };
    std::vector<int64_t> generated{token = likeliest()};
    arguments.assign({packs[4], packs[5], row.data(), &place, bias.data()});
    for (auto &buffer : prefix)
      arguments.push_back(buffer.data());
    for (auto &buffer : before)
      arguments.push_back(buffer.data());
    arguments.push_back(logits.data());
    for (auto &buffer : own)
      arguments.push_back(buffer.data());
    for (unsigned step = 0; step < GENERATED - 1 && token != EOS; ++step) {
      place = position + step;
      memcpy(row.data(), rows + token * HIDDEN, HIDDEN * sizeof(float));
      if (decode(arguments.data()))
        throw std::runtime_error("Decode program failed");
      for (unsigned i = 0; i < 2 * LAYERS; ++i)
        memcpy(before[i].data() + step * HEAD_DIM, own[i].data(),
               HEAD_DIM * sizeof(float));
      bias[PREFIX_KEYS + step] = 0.0f;
      generated.push_back(token = likeliest());
    }
    const uint64_t cycles = read_counter() - first;
    const double elapsed = (read_time_ns() - begin) * 1e-9;
    printf("Inference cycles: %lu\n", static_cast<unsigned long>(cycles));
    // The tokens against PyTorch (generated.data), as baremetal.cpp reports
    // them: its tokens up to the EOS.
    if (!config.referencePath.empty()) {
      std::vector<int64_t> expected(GENERATED);
      std::ifstream golden;
      golden.exceptions(std::ios::failbit | std::ios::badbit);
      golden.open(config.referencePath, std::ios::binary);
      golden.read(reinterpret_cast<char *>(expected.data()),
                  expected.size() * sizeof(int64_t));
      unsigned different = 0, length = 0;
      while (length < GENERATED && expected[length] != 0)
        ++length;
      const unsigned count = generated.size();
      for (unsigned i = 0; i < count && i < length; ++i)
        different += generated[i] != expected[i];
      printf("tokens: %u generated, %u in PyTorch, %u of the first %u differ\n",
             count, length, different, count < length ? count : length);
    }
    std::cout << "Tokens: " << generated.size()
              << "\nInference time: " << elapsed << "s\n";
    if (!config.outputPath.empty()) {
      std::ofstream file;
      file.exceptions(std::ios::failbit | std::ios::badbit);
      file.open(config.outputPath, std::ios::binary);
      file.write(reinterpret_cast<const char *>(generated.data()),
                 generated.size() * sizeof(int64_t));
    }
  }
};
} // namespace
extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new Pi0FastRunner();
}
extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

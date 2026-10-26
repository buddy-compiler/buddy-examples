#include <algorithm>
#include <array>
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <instances.h>
#include <iostream>
#include <linux.h>
#include <memory>
#include <params.h>
#include <stream.h>
#include <workspace.h>

extern "C" void _mlir_ciface_forward(MemRef<float, 2> *, MemRef<float, 1> *,
                                     MemRef<int8_t, 1> *, MemRef<int64_t, 2> *,
                                     MemRef<int64_t, 2> *, MemRef<int64_t, 2> *,
                                     MemRef<int64_t, 2> *);

namespace {
template <typename T> MemRef<T, 1> load(const std::string &path) {
  std::ifstream file;
  file.exceptions(std::ios::failbit | std::ios::badbit);
  file.open(path, std::ios::binary | std::ios::ate);
  size_t bytes = file.tellg();
  if (bytes % sizeof(T))
    throw std::runtime_error("Invalid parameter byte count");
  MemRef<T, 1> result(std::vector<size_t>{bytes / sizeof(T)});
  file.seekg(0);
  file.read(reinterpret_cast<char *>(result.getData()), bytes);
  return result;
}
struct Call {
  MemRef<float, 1> &parameters;
  MemRef<int8_t, 1> &weights;
  MemRef<int64_t, 2> &ids, &types, &mask, &positions;
  MemRef<float, 2> output{std::vector<size_t>{1, 6}, false, 0};
  std::array<float, 6> logits;
  std::exception_ptr error;
};
void forward(void *argument) {
  auto &call = *static_cast<Call *>(argument);
  _mlir_ciface_forward(&call.output, &call.parameters, &call.weights, &call.ids,
                       &call.types, &call.mask, &call.positions);
}
int execute(void *, void *argument) {
  auto &call = *static_cast<Call *>(argument);
  try {
    constexpr size_t bytes = size_t(1) << 27;
    std::unique_ptr<void, decltype(&std::free)> arena(aligned_alloc(64, bytes),
                                                      std::free);
    if (!arena)
      throw std::bad_alloc();
    workspace_init(arena.get(), bytes);
    workspace_begin(arena.get(), bytes);
    stream_run(THREAD_SIGNATURE, forward, &call);
    std::copy_n(call.output.getData(), call.logits.size(), call.logits.begin());
    workspace_free(call.output.release());
    return 0;
  } catch (...) {
    call.error = std::current_exception();
    return 1;
  }
}
} // namespace
class Runner final : public buddy::runtime::InferenceRunner {
public:
  void run(const buddy::runtime::RunConfig &config) override {
    const auto manifest =
        buddy::runtime::ModelManifest::loadFromRax(config.raxPath);
    auto parameters = load<float>(manifest.weightPaths.at(0));
    auto weights = load<int8_t>(manifest.weightPaths.at(1));
    buddy::Text<int64_t, 2> input(config.prompt);
    input.tokenizeBert(manifest.vocabPath, 32, true, true);
    MemRef<int64_t, 2> types({1, 32}, 0), mask({1, 32}, 0),
        positions({1, 32}, 0);
    for (size_t i = 0; i < 32; ++i) {
      positions.getData()[i] = i;
      mask.getData()[i] = i < input.getTokenCnt();
    }
    Call call{parameters, weights, input, types, mask, positions};
    const auto start = std::chrono::steady_clock::now();
    if (runtime_linux_job(bb_cores[0].hart, bb_cores[0].signature, execute,
                          &call))
      std::rethrow_exception(call.error);
    const size_t prediction =
        std::max_element(call.logits.begin(), call.logits.end()) -
        call.logits.begin();
    constexpr const char *labels[] = {"sadness", "joy",  "love",
                                      "anger",   "fear", "surprise"};
    double total = 0;
    for (auto logit : call.logits)
      total += std::exp(logit - call.logits[prediction]);
    std::cout << "Classification: " << labels[prediction]
              << "\nProbability: " << 1.0 / total << '\n';
    if (!config.suppressStats)
      std::cout << "Inference time: "
                << std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start)
                       .count()
                << "s\n";
  }
};

extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new Runner();
}
extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

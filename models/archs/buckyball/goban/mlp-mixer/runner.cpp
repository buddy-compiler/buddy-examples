#include "testutils.h"
#include <algorithm>
#include <buddy/Core/Container.h>
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <instances.h>
#include <iomanip>
#include <iostream>
#include <linux.h>
#include <memory>
#include <mlir/ExecutionEngine/CRunnerUtils.h>
#include <numeric>
#include <params.h>
#include <stdexcept>
#include <stream.h>
#include <string>
#include <vector>
#include <workspace.h>

using Matrix = std::vector<float>;

extern "C" void _mlir_ciface_forward(StridedMemRefType<float, 2> *,
                                     MemRef<float, 1> *, MemRef<int8_t, 1> *,
                                     StridedMemRefType<float, 2> *);

template <typename T> MemRef<T, 1> load(const std::string &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  stream.exceptions(std::ios::failbit | std::ios::badbit);
  size_t bytes = stream.tellg();
  if (bytes % sizeof(T))
    throw std::runtime_error("invalid tensor size: " + path);
  MemRef<T, 1> result(std::vector<size_t>{bytes / sizeof(T)});
  stream.seekg(0);
  stream.read(reinterpret_cast<char *>(result.getData()), bytes);
  return result;
}

struct Inference {
  MemRef<float, 1> *parameters;
  MemRef<int8_t, 1> *weights;
  StridedMemRefType<float, 2> input, output;
};

static void infer(void *argument) {
  auto &call = *static_cast<Inference *>(argument);
  _mlir_ciface_forward(&call.output, call.parameters, call.weights,
                       &call.input);
}

struct Run {
  std::string tensor, parameters, weights;
  bool suppressStats;
  std::exception_ptr error;
};

// Runs on the control core; linear layers stream their Ball work to its
// threads.
static int execute(void *, void *argument) {
  auto &call = *static_cast<Run *>(argument);
  try {
    auto input = load<float>(static_cast<Run *>(argument)->tensor);
    if (input.getSize() != 196 * 768)
      throw std::runtime_error("expected 196x768 patches");
    auto params = load<float>(call.parameters);
    auto weights = load<int8_t>(call.weights);
    constexpr size_t workspaceBytes = size_t(1) << 24;
    std::unique_ptr<void, decltype(&std::free)> arena(
        aligned_alloc(64, workspaceBytes), std::free);
    void *workspace = arena.get();
    if (!workspace)
      throw std::bad_alloc();
    workspace_init(workspace, workspaceBytes);
    std::fill_n(static_cast<float *>(workspace), workspaceBytes / sizeof(float),
                0.0f);
    workspace_begin(workspace, workspaceBytes);
    uint64_t blockCounters[12];
    uint64_t start = read_counter();
    Inference inference{
        &params,
        &weights,
        {input.getData(), input.getData(), 0, {196, 768}, {768, 1}},
        {}};
    stream_run(THREAD_SIGNATURE, infer, &inference);
    Matrix logits(inference.output.data + inference.output.offset,
                  inference.output.data + inference.output.offset + 1000);
    workspace_free(inference.output.basePtr);
    uint64_t elapsedCounter = read_counter() - start;
    for (float element : logits)
      if (!std::isfinite(element))
        throw std::runtime_error("Mixer produced non-finite logits");
    const auto prediction =
        std::max_element(logits.begin(), logits.end()) - logits.begin();
    std::cout << "Classification Index: " << prediction << '\n';
    if (!call.suppressStats)
      std::cout << "Counter: " << elapsedCounter << '\n';
    return 0;
  } catch (...) {
    call.error = std::current_exception();
    return 1;
  }
}

class Runner final : public buddy::runtime::InferenceRunner {
public:
  void run(const buddy::runtime::RunConfig &config) override {
    const auto manifest =
        buddy::runtime::ModelManifest::loadFromRax(config.raxPath);
    Run call{config.imagePath, manifest.weightPaths.at(0),
             manifest.weightPaths.at(1), config.suppressStats};
    if (runtime_linux_job(bb_cores[0].hart, bb_cores[0].signature, execute,
                          &call))
      std::rethrow_exception(call.error);
  }
};

extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new Runner();
}
extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

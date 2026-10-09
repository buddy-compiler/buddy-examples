#include "ffn/params.h"
#include "talker.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <runtime.h>

using namespace TalkerParams;
struct Routing {
  Matrix hidden;
  Tokens selected;
  Matrix scores, shared;
};
extern "C" void _mlir_ciface_forward_talker_prefill_router(Routing *, Floats *,
                                                           Matrix *);
extern "C" void _mlir_ciface_forward_talker_decode_router(Routing *, Floats *,
                                                          Matrix *);
using Kernel = void (*)(Matrix *, Matrix *, Bytes *, Bytes *, Bytes *);
#define EXPERT_KERNEL(N)                                                       \
  extern "C" void _mlir_ciface_subgraph_talker_expert_##N(                     \
      Matrix *, Matrix *, Bytes *, Bytes *, Bytes *);
EXPERT_KERNEL(1)
EXPERT_KERNEL(2)
EXPERT_KERNEL(4)
EXPERT_KERNEL(8)
EXPERT_KERNEL(16)
#undef EXPERT_KERNEL
extern "C" void _mlir_ciface_subgraph_talker_prefill_shared(Matrix *, Matrix *,
                                                            Bytes *, Bytes *,
                                                            Bytes *);
extern "C" void _mlir_ciface_subgraph_talker_decode_shared(Matrix *, Matrix *,
                                                           Bytes *, Bytes *,
                                                           Bytes *);

namespace {
struct Job {
  Matrix input, output;
  View<int8_t, 1> weight0, weight1, weight2;
  std::vector<size_t> slots;
  Kernel kernel;
  task *pending;
  Job(size_t rows, int8_t *packed, const std::array<size_t, 3> &sizes,
      Kernel run)
      : input({rows, width}, 0.0f), output({rows, width}, false, 0),
        weight0(packed, {sizes[0]}), weight1(packed + sizes[0], {sizes[1]}),
        weight2(packed + sizes[0] + sizes[1], {sizes[2]}), kernel(run) {}
};
void run(void *argument) {
  auto &job = *static_cast<Job *>(argument);
  job.kernel(&job.output, &job.input, &job.weight0, &job.weight1, &job.weight2);
}
} // namespace

void Talker::experts(Matrix &hidden, size_t count, size_t layer) {
  size_t length = count == 1 ? 1 : prefill;
  size_t first = 2 * groups + 4 + layer * (expertCount + 3);
  auto [fp, packed] = parameters(first + 1);
  Routing routed{
      Matrix({length, width}, false, 0), Tokens({length, topK}, false, 0),
      Matrix({length, topK}, false, 0), Matrix({length, 1}, false, 0)};
  auto route = count == 1 ? _mlir_ciface_forward_talker_decode_router
                          : _mlir_ciface_forward_talker_prefill_router;
  route(&routed, &fp, &hidden);
  std::vector<std::vector<size_t>> assigned(expertCount);
  for (size_t slot = 0; slot < count * topK; ++slot)
    assigned.at(routed.selected[slot]).push_back(slot);
  auto [shared_fp, shared_packed] = parameters(first + 2);
  Job shared(length, shared_packed.getData(),
             {sharedWeight0, sharedWeight1, sharedWeight2},
             count == 1 ? _mlir_ciface_subgraph_talker_decode_shared
                        : _mlir_ciface_subgraph_talker_prefill_shared);
  std::copy_n(routed.hidden.getData(), length * width, shared.input.getData());
  shared.pending = task_submit(CORE_SIGNATURE, run, &shared);
  std::vector<std::unique_ptr<Job>> jobs;
  for (size_t expert = 0; expert < expertCount; ++expert) {
    const auto &slots = assigned[expert];
    if (slots.empty())
      continue;
    size_t rows;
    Kernel kernel;
    if (slots.size() == 1) {
      rows = 1;
      kernel = _mlir_ciface_subgraph_talker_expert_1;
    } else if (slots.size() <= 2) {
      rows = 2;
      kernel = _mlir_ciface_subgraph_talker_expert_2;
    } else if (slots.size() <= 4) {
      rows = 4;
      kernel = _mlir_ciface_subgraph_talker_expert_4;
    } else if (slots.size() <= 8) {
      rows = 8;
      kernel = _mlir_ciface_subgraph_talker_expert_8;
    } else {
      rows = prefill;
      kernel = _mlir_ciface_subgraph_talker_expert_16;
    }
    auto [ef, ep] = parameters(first + 3 + expert);
    auto job = std::make_unique<Job>(
        rows, ep.getData(),
        std::array<size_t, 3>{expertWeight0, expertWeight1, expertWeight2},
        kernel);
    job->slots = slots;
    for (size_t index = 0; index < slots.size(); ++index)
      std::copy_n(routed.hidden.getData() + slots[index] / topK * width, width,
                  job->input.getData() + index * width);
    job->pending = task_submit(CORE_SIGNATURE, run, job.get());
    jobs.push_back(std::move(job));
  }
  std::vector<float> partials(count * topK * width);
  for (auto &job : jobs) {
    if (task_wait(job->pending))
      throw std::runtime_error("Talker expert task failed");
    for (size_t row = 0; row < job->slots.size(); ++row)
      std::copy_n(job->output.getData() + row * width, width,
                  partials.data() + job->slots[row] * width);
    workspace_free(job->output.release());
  }
  if (task_wait(shared.pending))
    throw std::runtime_error("Talker shared expert task failed");
  for (size_t token = 0; token < count; ++token)
    for (size_t column = 0; column < width; ++column) {
      float sum = 0;
      for (size_t choice = 0; choice < topK; ++choice)
        sum = std::fma(routed.scores[token * topK + choice],
                       partials[(token * topK + choice) * width + column], sum);
      sum += routed.shared[token] * shared.output[token * width + column];
      hidden[token * width + column] += sum;
    }
  workspace_free(shared.output.release());
  workspace_free(routed.hidden.release());
  workspace_free(routed.selected.release());
  workspace_free(routed.scores.release());
  workspace_free(routed.shared.release());
}

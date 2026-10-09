#include "ffn/params.h"
#include "state.h"
#include <algorithm>
#include <runtime.h>

extern "C" void _mlir_ciface_forward_prefill_router(RouterResult *, Floats *,
                                                    Matrix *);
extern "C" void _mlir_ciface_forward_decode_router(RouterResult *, Floats *,
                                                   Matrix *);
using ExpertKernel = void (*)(Matrix *, Matrix *, Bytes *, Bytes *, Bytes *);
extern "C" void _mlir_ciface_subgraph_expert(Matrix *, Matrix *, Bytes *,
                                             Bytes *, Bytes *);
extern "C" void _mlir_ciface_subgraph_expert_2(Matrix *, Matrix *, Bytes *,
                                               Bytes *, Bytes *);
extern "C" void _mlir_ciface_subgraph_expert_4(Matrix *, Matrix *, Bytes *,
                                               Bytes *, Bytes *);
extern "C" void _mlir_ciface_subgraph_expert_8(Matrix *, Matrix *, Bytes *,
                                               Bytes *, Bytes *);
extern "C" void _mlir_ciface_subgraph_expert_prefill(Matrix *, Matrix *,
                                                     Bytes *, Bytes *, Bytes *);

struct ExpertJob {
  Matrix input, output;
  View<int8_t, 1> weight0, weight1, weight2;
  std::vector<size_t> slots;
  ExpertKernel kernel;
  task *pending;

  ExpertJob(size_t rows, int8_t *weights, const std::vector<size_t> &assigned,
            ExpertKernel entry)
      : input({rows, hiddenSize}, 0.0f), output({rows, hiddenSize}, false, 0),
        weight0(weights, {expertWeight0}),
        weight1(weights + expertWeight0, {expertWeight1}),
        weight2(weights + expertWeight0 + expertWeight1, {expertWeight2}),
        slots(assigned), kernel(entry) {}
};

static void execute_expert(void *argument) {
  auto &job = *static_cast<ExpertJob *>(argument);
  job.kernel(&job.output, &job.input, &job.weight0, &job.weight1, &job.weight2);
}

void Thinker::router(size_t count, size_t layer) {
  size_t length = count == 1 ? 1 : prefillLength;
  Matrix hidden({length, hiddenSize}, 0.0f);
  read_values(hidden.getData(), count * hiddenSize);
  auto fp = float_parameters(layer_region(layer) + 1);
  RouterResult result{Matrix({length, hiddenSize}, false, 0),
                      Tokens({length, topK}, false, 0),
                      Matrix({length, topK}, false, 0)};
  auto run = count == 1 ? _mlir_ciface_forward_decode_router
                        : _mlir_ciface_forward_prefill_router;
  run(&result, &fp, &hidden);
  write_values(result.hidden.getData(), count * hiddenSize);
  write_values(result.experts.getData(), count * topK);
  write_values(result.scores.getData(), count * topK);
  workspace_free(result.hidden.release());
  workspace_free(result.experts.release());
  workspace_free(result.scores.release());
}

void Thinker::experts(size_t count, size_t layer) {
  size_t first_expert = layer_region(layer) + 2;
  Matrix hidden({count, hiddenSize}), scores({count, topK}),
      output({count, hiddenSize}, 0.0f);
  Tokens selected({count, topK});
  read_values(hidden.getData(), count * hiddenSize);
  read_values(selected.getData(), count * topK);
  read_values(scores.getData(), count * topK);
  std::vector<std::vector<size_t>> assigned(expertsCount);
  for (size_t slot = 0; slot < count * topK; ++slot) {
    auto expert = selected[slot];
    if (expert < 0 || size_t(expert) >= expertsCount)
      throw std::runtime_error("invalid expert index");
    assigned[expert].push_back(slot);
  }
  Matrix partials({count * topK, hiddenSize});
  std::vector<std::unique_ptr<ExpertJob>> jobs;
  for (size_t expert = 0; expert < expertsCount; ++expert) {
    const auto &slots = assigned[expert];
    if (slots.empty())
      continue;
    size_t length = slots.size();
    auto run = _mlir_ciface_subgraph_expert;
    if (length > 8) {
      length = prefillLength;
      run = _mlir_ciface_subgraph_expert_prefill;
    } else if (length > 4) {
      length = 8;
      run = _mlir_ciface_subgraph_expert_8;
    } else if (length > 2) {
      length = 4;
      run = _mlir_ciface_subgraph_expert_4;
    } else if (length > 1) {
      length = 2;
      run = _mlir_ciface_subgraph_expert_2;
    }
    auto weights = byte_parameters(first_expert + expert);
    auto job =
        std::make_unique<ExpertJob>(length, weights.getData(), slots, run);
    for (size_t row = 0; row < slots.size(); ++row)
      std::copy_n(hidden.getData() + slots[row] / topK * hiddenSize, hiddenSize,
                  job->input.getData() + row * hiddenSize);
    job->pending = task_submit(CORE_SIGNATURE, execute_expert, job.get());
    jobs.push_back(std::move(job));
  }
  for (auto &job : jobs) {
    if (task_wait(job->pending))
      throw std::runtime_error("expert task failed");
    for (size_t row = 0; row < job->slots.size(); ++row)
      std::copy_n(job->output.getData() + row * hiddenSize, hiddenSize,
                  partials.getData() + job->slots[row] * hiddenSize);
    workspace_free(job->output.release());
  }
  for (size_t token = 0; token < count; ++token) {
    for (size_t choice = 0; choice < topK; ++choice) {
      for (size_t i = 0; i < hiddenSize; ++i)
        output[token * hiddenSize + i] +=
            partials[(token * topK + choice) * hiddenSize + i] *
            scores[token * topK + choice];
    }
  }
  write_values(output.getData(), count * hiddenSize);
}

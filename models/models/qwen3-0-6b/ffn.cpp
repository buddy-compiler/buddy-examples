#include "ffn.h"
#include <params.h>
#include <runtime.h>
#include <cstring>
#include <stdexcept>

struct ExpandTask {
  FfnExpandKernel kernel;
  Hidden *result, *input;
  Floats *norm;
  FfnWeightView gate, up;
};
struct DownTask {
  FfnDownKernel kernel;
  Hidden *result, *input;
  FfnWeightView weights;
};

static void expandTask(void *argument) {
  auto &task = *static_cast<ExpandTask *>(argument);
  task.kernel(task.result, task.input, task.norm, &task.gate, &task.up);
}
static void downTask(void *argument) {
  auto &task = *static_cast<DownTask *>(argument);
  task.kernel(task.result, task.input, &task.weights);
}

void runFfnExpand(const FfnKernels &kernels, Hidden *result, Floats *norm, Bytes *weights, Hidden *input) {
  size_t length = input->getSizes()[1], width = kernels.intermediate / 2;
  Hidden shards[3] = {Hidden({1, length, kernels.expandWidths[0]}, false, 0),
                      Hidden({1, length, kernels.expandWidths[1]}, false, 0),
                      Hidden({1, length, kernels.expandWidths[2]}, false, 0)};
  std::array<ExpandTask, 3> jobs;
  std::array<task *, 3> tasks;
  size_t begin = 0;
  for (size_t index = 0; index < 3; ++index) {
    auto gate = kernels.gateWeights[index], up = kernels.upWeights[index];
    jobs[index] = {kernels.expand[index], &shards[index], input, norm,
                   {weights->getData(), weights->getData(), intptr_t(gate.offset), intptr_t(gate.bytes), 1},
                   {weights->getData(), weights->getData(), intptr_t(up.offset), intptr_t(up.bytes), 1}};
    tasks[index] = task_submit(CORE_SIGNATURE, expandTask, &jobs[index]);
    begin += kernels.expandWidths[index];
  }
  std::unique_ptr<float> allocation(static_cast<float *>(workspace_alloc(length * width * sizeof(float))));
  intptr_t shape[] = {1, intptr_t(length), intptr_t(width)};
  *result = Hidden(allocation, shape);
  begin = 0;
  for (size_t index = 0; index < 3; ++index) {
    if (task_wait(tasks[index]))
      throw std::runtime_error("FFN expand task failed");
    for (size_t token = 0; token < length; ++token)
      std::memcpy(result->getData() + token * width + begin,
                  shards[index].getData() + token * kernels.expandWidths[index],
                  kernels.expandWidths[index] * sizeof(float));
    workspace_free(shards[index].release());
    begin += kernels.expandWidths[index];
  }
}

void runFfnDown(const FfnKernels &kernels, Hidden *result, Bytes *weights, Hidden *input) {
  // Each worker consumes the complete original intermediate, preserving K accumulation order.
  size_t length = input->getSizes()[1], width = kernels.hidden / 2;
  Hidden shards[3] = {Hidden({1, length, kernels.downWidths[0]}, false, 0),
                      Hidden({1, length, kernels.downWidths[1]}, false, 0),
                      Hidden({1, length, kernels.downWidths[2]}, false, 0)};
  std::array<DownTask, 3> jobs;
  std::array<task *, 3> tasks;
  size_t begin = 0;
  for (size_t index = 0; index < 3; ++index) {
    auto matrix = kernels.downWeights[index];
    jobs[index] = {kernels.down[index], &shards[index], input,
                   {weights->getData(), weights->getData(), intptr_t(matrix.offset), intptr_t(matrix.bytes), 1}};
    tasks[index] = task_submit(CORE_SIGNATURE, downTask, &jobs[index]);
    begin += kernels.downWidths[index];
  }
  std::unique_ptr<float> allocation(static_cast<float *>(workspace_alloc(length * width * sizeof(float))));
  intptr_t shape[] = {1, intptr_t(length), intptr_t(width)};
  *result = Hidden(allocation, shape);
  begin = 0;
  for (size_t index = 0; index < 3; ++index) {
    if (task_wait(tasks[index]))
      throw std::runtime_error("FFN down task failed");
    for (size_t token = 0; token < length; ++token)
      std::memcpy(result->getData() + token * width + begin,
                  shards[index].getData() + token * kernels.downWidths[index],
                  kernels.downWidths[index] * sizeof(float));
    workspace_free(shards[index].release());
    begin += kernels.downWidths[index];
  }
}

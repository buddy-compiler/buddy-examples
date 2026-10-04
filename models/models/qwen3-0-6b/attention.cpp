#include "attention.h"
#include <params.h>
#include <runtime.h>
#include <stdexcept>

struct BodyTask {
  AttentionBodyKernel kernel;
  AttentionBodyResult *result;
  Hidden *hidden;
  AttentionFloatView norm, queryNorm, keyNorm, frequencies;
  AttentionWeightView query, key, value;
  Positions *positions;
  Cache *keys, *values;
};
struct ProjectionTask {
  AttentionProjectionKernel kernel;
  Hidden *result, *context;
  AttentionWeightView weights;
};

static void bodyTask(void *argument) {
  auto &task = *static_cast<BodyTask *>(argument);
  task.kernel(task.result, task.hidden, &task.norm, &task.query, &task.queryNorm, &task.key, &task.keyNorm,
              &task.value, task.positions, &task.frequencies, task.keys, task.values);
}
static void projectionTask(void *argument) {
  auto &task = *static_cast<ProjectionTask *>(argument);
  task.kernel(task.result, task.context, &task.weights);
}

void runAttentionBody(const AttentionKernels &kernels, AttentionBodyResult *result, Floats *floats, Bytes *weights,
                      Hidden *hidden, Cache *keys, Cache *values, Positions *positions) {
  float *parameters = floats->getData();
  int8_t *packed = weights->getData();
  size_t head = kernels.headSize;
  BodyTask job{kernels.body, result, hidden,
               {parameters, parameters, intptr_t(head / 2 + head), hidden->getSizes()[2], 1},
               {parameters, parameters, intptr_t(head / 2 + head + hidden->getSizes()[2]), intptr_t(head), 1},
               {parameters, parameters, intptr_t(head / 2), intptr_t(head), 1},
               {parameters, parameters, 0, intptr_t(head / 2), 1},
               {packed, packed, intptr_t(kernels.query.offset), intptr_t(kernels.query.bytes), 1},
               {packed, packed, intptr_t(kernels.key.offset), intptr_t(kernels.key.bytes), 1},
               {packed, packed, intptr_t(kernels.value.offset), intptr_t(kernels.value.bytes), 1},
               positions, keys, values};
  if (task_wait(task_submit(CORE_SIGNATURE, bodyTask, &job)))
    throw std::runtime_error("attention body task failed");
}

void runAttentionProjection(const AttentionKernels &kernels, Hidden *result, Bytes *weights, Hidden *context) {
  // Each output row retains the complete original group context and its K accumulation order.
  int8_t *packed = weights->getData();
  ProjectionTask job{kernels.projection, result, context,
                     {packed, packed, intptr_t(kernels.projectionWeights.offset), intptr_t(kernels.projectionWeights.bytes), 1}};
  if (task_wait(task_submit(CORE_SIGNATURE, projectionTask, &job)))
    throw std::runtime_error("attention projection task failed");
}

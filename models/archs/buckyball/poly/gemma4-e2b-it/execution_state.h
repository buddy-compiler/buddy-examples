#pragma once
#include "execution.h"
#include "layer_execution.h"

struct Execution::State {
  Request request;
  std::unique_ptr<LayerExecution> execution;
  std::span<const float> output;
  size_t position = 0;
  bool prefilled = false, closed = false;
};

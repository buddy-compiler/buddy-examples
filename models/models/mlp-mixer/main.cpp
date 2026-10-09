#include "testutils.h"
#include <algorithm>
#include <buddy/Core/Container.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mlir/ExecutionEngine/CRunnerUtils.h>
#include <numeric>
#include <params.h>
#include <runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

using Matrix = std::vector<float>;

#include "mixer-parameters.h"

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

struct Shard {
  size_t layer, rows;
  MemRef<float, 2> *input;
  float *output;
  MemRef<float, 1> *params;
  MemRef<int8_t, 1> *weights;
};

static void execute_shard(void *argument) {
  auto &shard = *static_cast<Shard *>(argument);
  const auto *shape = shapes[shard.layer];
  StridedMemRefType<float, 2> input = {shard.input->getData(),
                                       shard.input->getData(),
                                       0,
                                       {intptr_t(shape[3]), intptr_t(shape[1])},
                                       {intptr_t(shape[1]), 1}};
  StridedMemRefType<float, 2> output;
  linears[shard.layer](&output, shard.params, shard.weights, &input);
  for (size_t row = 0; row < shard.rows; ++row)
    std::copy_n(output.data + output.offset + row * output.strides[0], shape[2],
                shard.output + row * shape[2]);
  workspace_free(output.basePtr);
}

static Matrix linear(size_t layer, const Matrix &input,
                     MemRef<float, 1> &params, MemRef<int8_t, 1> &weights) {
  const auto *shape = shapes[layer];
  Matrix result(shape[0] * shape[2]);
  Shard shards[4];
  task *tasks[4];
  std::vector<MemRef<float, 2>> inputs;
  inputs.reserve(4);
  size_t count = 0;
  for (size_t row = 0; row < shape[0]; row += shape[3]) {
    inputs.emplace_back(std::vector<size_t>{shape[3], shape[1]});
    std::fill_n(inputs.back().getData(), inputs.back().getSize(), 0.0f);
    size_t rows = std::min(shape[3], shape[0] - row);
    std::copy_n(input.data() + row * shape[1], rows * shape[1],
                inputs.back().getData());
    shards[count] = {layer,          std::min(shape[3], shape[0] - row),
                     &inputs.back(), result.data() + row * shape[2],
                     &params,        &weights};
    ++count;
  }
  for (size_t index = 0; index < count; ++index)
    tasks[index] = task_submit(CORE_SIGNATURE, execute_shard, &shards[index]);
  for (size_t index = 0; index < count; ++index)
    if (task_wait(tasks[index]))
      throw std::runtime_error("Mixer shard failed");
  return result;
}

static Matrix normalize(const Matrix &input, size_t norm, const float *params) {
  Matrix result(input.size());
  for (size_t row = 0; row < input.size() / 768; ++row) {
    double sum = 0.0, variance = 0.0;
    for (size_t col = 0; col < 768; ++col)
      sum += input[row * 768 + col];
    float mean = sum / 768;
    for (size_t col = 0; col < 768; ++col) {
      double centered = input[row * 768 + col] - mean;
      variance += centered * centered;
    }
    float scale = 1.0f / std::sqrt(float(variance / 768) + 1e-6f);
    for (size_t col = 0; col < 768; ++col)
      result[row * 768 + col] = (input[row * 768 + col] - mean) * scale *
                                    params[norms[norm][0] + col] +
                                params[norms[norm][1] + col];
  }
  return result;
}

static Matrix transpose(const Matrix &input, size_t rows, size_t columns) {
  Matrix result(input.size());
  for (size_t row = 0; row < rows; ++row)
    for (size_t col = 0; col < columns; ++col)
      result[col * rows + row] = input[row * columns + col];
  return result;
}
static void gelu(Matrix &value) {
  for (float &element : value)
    element = 0.5f * element * (1.0f + std::erf(element * 0.7071067811865475f));
}
static void residual(Matrix &value, const Matrix &other) {
  for (size_t index = 0; index < value.size(); ++index)
    value[index] += other[index];
}

int main(int argc, char **argv) {
  if (argc != 3 || std::string(argv[1]) != "--tensor")
    throw std::runtime_error(
        "expected --tensor input.f32 with 196x768 prepared patches");
  auto input = load<float>(argv[2]);
  if (input.getSize() != 196 * 768)
    throw std::runtime_error("expected 196x768 patches");
  auto params = load<float>("mixer.payload/params.f32");
  auto weights = load<int8_t>("mixer.payload/weights.bin");
  runtime_init(1024 * 1024);
  if (core_count() != 4)
    throw std::runtime_error(
        "Mixer recipe requires four homogeneous compute cores");
  constexpr size_t workspaceBytes = size_t(1) << 24;
  void *workspace = aligned_alloc(64, workspaceBytes);
  if (!workspace)
    throw std::bad_alloc();
  workspace_init(workspace, workspaceBytes);
  std::fill_n(static_cast<float *>(workspace), workspaceBytes / sizeof(float),
              0.0f);
  workspace_begin(workspace, workspaceBytes);
  Matrix pixels(input.getData(), input.getData() + input.getSize());
  uint64_t blockCounters[12];
  uint64_t start = read_counter();
  Matrix value = linear(0, pixels, params, weights);
  for (size_t block = 0; block < 12; ++block) {
    uint64_t blockStart = read_counter();
    Matrix mixed =
        transpose(normalize(value, 2 * block, params.getData()), 196, 768);
    mixed = linear(1 + 4 * block, mixed, params, weights);
    gelu(mixed);
    mixed = linear(2 + 4 * block, mixed, params, weights);
    residual(value, transpose(mixed, 768, 196));
    mixed =
        linear(3 + 4 * block, normalize(value, 2 * block + 1, params.getData()),
               params, weights);
    gelu(mixed);
    mixed = linear(4 + 4 * block, mixed, params, weights);
    residual(value, mixed);
    blockCounters[block] = read_counter() - blockStart;
  }
  Matrix normalized = normalize(value, 24, params.getData());
  Matrix mean(768, 0.0f);
  for (size_t row = 0; row < 196; ++row)
    for (size_t col = 0; col < 768; ++col)
      mean[col] += normalized[row * 768 + col] / 196.0f;
  Matrix logits = linear(49, mean, params, weights);
  uint64_t elapsedCounter = read_counter() - start;
  for (float element : logits)
    if (!std::isfinite(element))
      throw std::runtime_error("Mixer produced non-finite logits");
  std::ofstream output("logits.f32", std::ios::binary);
  output.exceptions(std::ios::failbit | std::ios::badbit);
  output.write(reinterpret_cast<const char *>(logits.data()),
               logits.size() * sizeof(float));
  std::vector<size_t> order(1000);
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                    [&](size_t a, size_t b) { return logits[a] > logits[b]; });
  std::cout << "Counter (" << counter_name() << ") " << counter_field() << ": "
            << elapsedCounter << "\nCompute cores: " << core_count()
            << "\nWorkspace peak bytes: " << workspace_peak()
            << "\nClassification Index: " << order[0] << "\nTop5:";
  for (size_t index = 0; index < 5; ++index)
    std::cout << ' ' << order[index];
  std::cout << '\n';
  for (size_t block = 0; block < 12; ++block)
    std::cout << "Block " << block << " " << counter_field() << ": "
              << blockCounters[block] << '\n';
  for (size_t core = 1; core <= core_count(); ++core)
    std::cout << "Core " << core << " tasks: " << core_submissions(core)
              << '\n';
  std::cout << "LOGITS_F32_BEGIN\n" << std::hex << std::setfill('0');
  for (size_t index = 0; index < logits.size(); ++index) {
    uint32_t bits;
    std::memcpy(&bits, &logits[index], sizeof(bits));
    std::cout << std::setw(8) << bits << (index % 8 == 7 ? '\n' : ' ');
  }
  std::cout << "\nLOGITS_F32_END\n" << std::dec;
  free(workspace);
}

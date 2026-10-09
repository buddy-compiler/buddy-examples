#include "worker.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <runtime.h>
#include <sched.h>
#include <stdexcept>
#include <sys/mman.h>
#include <system_error>
#include <topology.h>

#include "qwen-parameters.h"

const ModelShape &modelShape() {
  static const ModelShape shape{hiddenSize,
                                layers,
                                kvHeads,
                                attentionContext,
                                attentionProjectionInput,
                                parts,
                                executionTiles,
                                ffnIntermediate,
                                headSize,
                                vocabulary,
                                prefillLength,
                                cacheLength};
  return shape;
}

struct Weights {
  ReadonlyMappedMemRef<float> floats;
  ReadonlyMappedMemRef<int8_t> bytes;
  Weights(const std::filesystem::path &root, Parameters spec)
      : floats(root / spec.directory / "params.f32", spec.floats,
               alignof(float)),
        bytes(root / spec.directory / "weights.bin", spec.bytes,
              spec.alignment) {}
};

template <typename T, size_t N> struct View : MemRef<T, N> {
  View(T *data, std::vector<size_t> shape) : MemRef<T, N>(shape, false, 0) {
    this->aligned = data;
  }
};

static void runWorker(Worker &worker, size_t rank, int cpu, int64_t *tokens,
                      const std::filesystem::path &directory) {
  cpu_set_t cpus;
  CPU_ZERO(&cpus);
  CPU_SET(cpu, &cpus);
  if (sched_setaffinity(0, sizeof(cpus), &cpus))
    throw std::system_error(errno, std::generic_category(), "pin tile worker");
  void *workspace = aligned_alloc(64, workspaceBytes);
  if (!workspace)
    throw std::bad_alloc();
  runtime_init(1024 * 1024);
  std::unique_ptr<void, void (*)(void *)> arena(workspace, [](void *memory) {
    runtime_shutdown();
    free(memory);
  });
  workspace_init(workspace, workspaceBytes);
  std::map<std::string, std::unique_ptr<Weights>> weights;
  auto load = [&](Parameters parameters) {
    if (!weights.contains(parameters.directory))
      weights.emplace(parameters.directory,
                      std::make_unique<Weights>(directory, parameters));
  };
  if (rank == 0) {
    load(prefill_embedding.parameters);
    load(decode_embedding.parameters);
  }
  load(output[rank].parameters);
  for (size_t layer = 0; layer < layers; ++layer) {
    load(prefill_attention[rank][layer].parameters);
    load(decode_attention[rank][layer].parameters);
    load(prefill_ffn[rank][layer].parameters);
    load(decode_ffn[rank][layer].parameters);
  }
  const size_t codes = kvHeads * cacheLength * headSize;
  const size_t scales = codes / 32;
  auto &command = *worker.command;
  std::unique_lock lock(command.mutex);
  command.ready = true;
  command.changed.notify_all();
  for (;;) {
    command.changed.wait(
        lock, [&] { return command.state == 1 || command.stopping; });
    if (command.state != 1 && command.stopping)
      break;
    lock.unlock();
    const auto operation = command.operation, count = command.count;
    const auto start = command.start, layer = command.layer;
    const bool prefill = start == 0;
    const size_t length = prefill ? prefillLength : 1;
    workspace_begin(workspace, workspaceBytes);
    size_t resultWidth;
    if (operation == 0) {
      View<int64_t, 2> input(tokens, {1, length});
      Hidden result({1, length, hiddenSize}, false, 0);
      const auto &entry = prefill ? prefill_embedding : decode_embedding;
      entry.run(&result, &weights.at(entry.parameters.directory)->bytes,
                &input);
      std::memcpy(worker.output, result.getData(),
                  count * hiddenSize * sizeof(float));
      workspace_free(result.release());
    } else if (operation == 6) {
      View<float, 3> hidden(worker.input, {1, length, hiddenSize});
      View<int8_t, 4> keyCodes(worker.keyCodes + layer * codes,
                               {1, kvHeads, cacheLength, headSize});
      View<int8_t, 4> keyScales(worker.keyScales + layer * scales,
                                {1, kvHeads, cacheLength, headSize / 32});
      View<int8_t, 4> valueCodes(worker.valueCodes + layer * codes,
                                 {1, kvHeads, cacheLength, headSize});
      View<int8_t, 4> valueScales(worker.valueScales + layer * scales,
                                  {1, kvHeads, cacheLength, headSize / 32});
      Positions positions({length});
      for (size_t token = 0; token < length; ++token)
        positions.getData()[token] = start + token;
      const auto &entry = prefill ? prefill_attention[rank][layer]
                                  : decode_attention[rank][layer];
      auto &parameters = *weights.at(entry.parameters.directory);
      AttentionBodyResult result{
          Cache({1, kvHeads, cacheLength, headSize}, false, 0),
          Cache({1, kvHeads, cacheLength, headSize / 32}, false, 0),
          Cache({1, kvHeads, cacheLength, headSize}, false, 0),
          Cache({1, kvHeads, cacheLength, headSize / 32}, false, 0),
          Hidden({1, length, attentionContext}, false, 0)};
      runAttentionBody(*entry.kernels, &result, &parameters.floats,
                       &parameters.bytes, &hidden, &keyCodes, &keyScales,
                       &valueCodes, &valueScales, &positions);
      std::memcpy(worker.output, result.context.getData(),
                  count * attentionContext * sizeof(float));
      Cache *produced[] = {&result.keyCodes, &result.keyScales,
                           &result.valueCodes, &result.valueScales};
      Cache *persistent[] = {&keyCodes, &keyScales, &valueCodes, &valueScales};
      for (size_t field = 0; field < 4; ++field) {
        const size_t width = persistent[field]->getSizes()[3];
        for (size_t head = 0; head < kvHeads; ++head) {
          size_t offset = (head * cacheLength + start) * width;
          std::memcpy(persistent[field]->getData() + offset,
                      produced[field]->getData() + offset, count * width);
        }
        workspace_free(produced[field]->release());
      }
      workspace_free(result.context.release());
    } else {
      const size_t inputWidth = operation == 5   ? ffnIntermediate
                                : operation == 7 ? attentionProjectionInput
                                                 : hiddenSize;
      View<float, 3> input(worker.input,
                           {1, operation == 1 ? 1 : length, inputWidth});
      resultWidth = operation == 1   ? vocabulary / executionTiles
                    : operation == 4 ? ffnIntermediate / 2
                                     : hiddenSize / 2;
      Hidden result({1, operation == 1 ? 1 : length, resultWidth}, false, 0);
      if (operation == 1) {
        auto &parameters = *weights.at(output[rank].parameters.directory);
        output[rank].run(&result, &parameters.floats, &parameters.bytes,
                         &input);
      } else if (operation == 7) {
        const auto &entry = prefill ? prefill_attention[rank][layer]
                                    : decode_attention[rank][layer];
        runAttentionProjection(*entry.kernels, &result,
                               &weights.at(entry.parameters.directory)->bytes,
                               &input);
      } else {
        const auto &entry =
            prefill ? prefill_ffn[rank][layer] : decode_ffn[rank][layer];
        auto &parameters = *weights.at(entry.parameters.directory);
        if (operation == 4)
          runFfnExpand(*entry.kernels, &result, &parameters.floats,
                       &parameters.bytes, &input);
        else if (operation == 5)
          runFfnDown(*entry.kernels, &result, &parameters.bytes, &input);
        else
          throw std::runtime_error("invalid model stage");
      }
      std::memcpy(worker.output, result.getData(),
                  count * resultWidth * sizeof(float));
      workspace_free(result.release());
    }
    lock.lock();
    command.state = 2;
    command.changed.notify_all();
  }
}

Execution::State::State(const std::filesystem::path &directory,
                        const std::vector<uint64_t> &tiles) {
  if (tiles.size() != executionTiles)
    throw std::runtime_error("input tile count differs from compiled model");
  const size_t inputBytes =
      prefillLength *
      std::max({hiddenSize, attentionProjectionInput, ffnIntermediate}) *
      sizeof(float);
  const size_t outputBytes =
      std::max(prefillLength * std::max({hiddenSize, attentionContext,
                                         ffnIntermediate / 2}),
               vocabulary / executionTiles) *
      sizeof(float);
  const size_t codeBytes = layers * kvHeads * cacheLength * headSize;
  const size_t scaleBytes = codeBytes / 32;
  storageBytes =
      256 + prefillLength * (sizeof(int64_t) + hiddenSize * sizeof(float)) +
      executionTiles * (sizeof(Command) + inputBytes + outputBytes + 256);
  storage = mmap(nullptr, storageBytes, PROT_READ | PROT_WRITE,
                 MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (storage == MAP_FAILED)
    throw std::system_error(errno, std::generic_category(),
                            "shared model storage");
  uintptr_t cursor = reinterpret_cast<uintptr_t>(storage);
  auto reserve = [&](size_t bytes) {
    cursor = (cursor + 63) & ~uintptr_t(63);
    void *result = reinterpret_cast<void *>(cursor);
    cursor += bytes;
    return result;
  };
  tokens = static_cast<int64_t *>(reserve(prefillLength * sizeof(int64_t)));
  hidden =
      static_cast<float *>(reserve(prefillLength * hiddenSize * sizeof(float)));
  workers.resize(executionTiles);
  for (auto &worker : workers) {
    worker.command = new (reserve(sizeof(Command))) Command{};
    worker.input = static_cast<float *>(reserve(inputBytes));
    worker.output = static_cast<float *>(reserve(outputBytes));
    for (int8_t **data : {&worker.keyCodes, &worker.valueCodes}) {
      *data =
          static_cast<int8_t *>(mmap(nullptr, codeBytes, PROT_READ | PROT_WRITE,
                                     MAP_SHARED | MAP_ANONYMOUS, -1, 0));
      if (*data == MAP_FAILED)
        throw std::system_error(errno, std::generic_category(),
                                "shared KV codes");
    }
    for (int8_t **scale : {&worker.keyScales, &worker.valueScales}) {
      *scale = static_cast<int8_t *>(mmap(nullptr, scaleBytes,
                                          PROT_READ | PROT_WRITE,
                                          MAP_SHARED | MAP_ANONYMOUS, -1, 0));
      if (*scale == MAP_FAILED)
        throw std::system_error(errno, std::generic_category(),
                                "shared KV scales");
    }
  }
  resetCache();
  for (size_t rank = 0; rank < executionTiles; ++rank) {
    const int cpu = controlCpu(tiles[rank]);
    workers[rank].thread = std::thread([this, rank, cpu, directory] {
      auto &worker = workers[rank];
      try {
        runWorker(worker, rank, cpu, tokens, directory);
      } catch (const std::exception &error) {
        std::lock_guard lock(worker.command->mutex);
        std::snprintf(worker.command->error, sizeof(worker.command->error),
                      "%s", error.what());
        worker.command->state = 4;
        worker.command->changed.notify_all();
      }
    });
  }
  for (auto &worker : workers) {
    std::unique_lock lock(worker.command->mutex);
    worker.command->changed.wait(lock, [&] {
      return worker.command->ready || worker.command->state == 4;
    });
    if (worker.command->state == 4) {
      lock.unlock();
      close();
    }
  }
}

Execution::State::~State() {
  const size_t codeBytes = layers * kvHeads * cacheLength * headSize;
  for (auto &worker : workers) {
    worker.command->~Command();
    munmap(worker.keyCodes, codeBytes);
    munmap(worker.valueCodes, codeBytes);
    munmap(worker.keyScales, codeBytes / 32);
    munmap(worker.valueScales, codeBytes / 32);
  }
  munmap(storage, storageBytes);
}
void Execution::State::close() {
  if (closed)
    throw std::runtime_error("model workers already stopped");
  for (auto &worker : workers) {
    std::lock_guard lock(worker.command->mutex);
    worker.command->stopping = true;
    worker.command->changed.notify_all();
  }
  for (auto &worker : workers)
    worker.thread.join();
  closed = true;
  for (auto &worker : workers)
    if (worker.command->state == 4)
      throw std::runtime_error(worker.command->error);
}
void Execution::State::submit(size_t rank, uint64_t operation, size_t count,
                              size_t start, size_t layer) {
  auto &command = *workers.at(rank).command;
  std::lock_guard lock(command.mutex);
  if (command.state == 4)
    throw std::runtime_error(command.error);
  if (command.state != 0 || command.stopping)
    throw std::runtime_error("worker is not idle");
  command.operation = operation;
  command.count = count;
  command.start = start;
  command.layer = layer;
  command.state = 1;
  command.changed.notify_all();
}
void Execution::State::wait(size_t rank) {
  auto &command = *workers.at(rank).command;
  std::unique_lock lock(command.mutex);
  command.changed.wait(
      lock, [&] { return command.state == 2 || command.state == 4; });
  if (command.state == 4)
    throw std::runtime_error(command.error);
  command.state = 0;
}

void Execution::State::resetCache() {
  const size_t codeBytes = layers * kvHeads * cacheLength * headSize;
  for (auto &worker : workers) {
    std::memset(worker.keyCodes, 0, codeBytes);
    std::memset(worker.valueCodes, 0, codeBytes);
    std::memset(worker.keyScales, 127, codeBytes / 32);
    std::memset(worker.valueScales, 127, codeBytes / 32);
  }
}

Execution::~Execution() noexcept(false) {
  if (!state->closed)
    state->close();
}
const Request &Execution::request() const { return state->request; }
std::span<const float> Execution::logits() const { return state->logitValues; }
void Execution::close() { state->close(); }

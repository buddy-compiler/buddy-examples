#include "worker.h"
#include "llama-parameters.h"
#include "weights.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <runtime.h>
#include <sched.h>
#include <stdexcept>
#include <system_error>
constexpr size_t executionTiles = parts;

const ModelShape &modelShape() {
  static const ModelShape shape{hiddenSize,     layers,        kvHeads,
                                hiddenSize,     hiddenSize,    parts,
                                executionTiles, hiddenSize,    headSize,
                                vocabulary,     prefillLength, cacheLength};
  return shape;
}

struct Weights {
  ReadonlyMappedMemRef<float> floats;
  ReadonlyMappedMemRef<int8_t> bytes;
  Weights(const std::filesystem::path &root, Parameters spec)
      : floats(root / spec.directory / "params.f32", spec.floats,
               alignof(float)),
        bytes(root / spec.directory / "weights.bin", spec.bytes, 16) {}
};

template <typename T, size_t N> struct View : MemRef<T, N> {
  View(T *data, std::vector<size_t> shape) : MemRef<T, N>(shape, false, 0) {
    this->aligned = data;
  }
};

void runWorker(Worker &worker, size_t rank, int cpu, int64_t *tokens,
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
  if (rank == 0)
    load(output.parameters);
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
    } else if (operation == 2) {
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
      AttentionResult result{
          Hidden({1, length, hiddenSize}, false, 0),
          Cache({1, kvHeads, cacheLength, headSize}, false, 0),
          Cache({1, kvHeads, cacheLength, headSize / 32}, false, 0),
          Cache({1, kvHeads, cacheLength, headSize}, false, 0),
          Cache({1, kvHeads, cacheLength, headSize / 32}, false, 0)};
      entry.run(&result, &parameters.floats, &parameters.bytes, &hidden,
                &keyCodes, &keyScales, &valueCodes, &valueScales, &positions);
      std::memcpy(worker.output, result.hidden.getData(),
                  count * hiddenSize * sizeof(float));
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
      workspace_free(result.hidden.release());
    } else if (operation == 1 || operation == 3) {
      View<float, 3> hidden(worker.input,
                            {1, operation == 1 ? 1 : length, hiddenSize});
      resultWidth = operation == 1 ? vocabulary : hiddenSize;
      Hidden result({1, operation == 1 ? 1 : length, resultWidth}, false, 0);
      const auto &entry = operation == 1 ? output
                                         : (prefill ? prefill_ffn[rank][layer]
                                                    : decode_ffn[rank][layer]);
      auto &parameters = *weights.at(entry.parameters.directory);
      entry.run(&result, &parameters.floats, &parameters.bytes, &hidden);
      std::memcpy(worker.output, result.getData(),
                  count * resultWidth * sizeof(float));
      workspace_free(result.release());
    } else
      throw std::runtime_error("invalid model stage");
    lock.lock();
    command.state = 2;
    command.changed.notify_all();
  }
}

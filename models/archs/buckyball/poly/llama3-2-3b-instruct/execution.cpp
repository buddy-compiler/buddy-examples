#include "execution_state.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <system_error>

Execution::State::State(const std::filesystem::path &directory,
                        const std::vector<uint64_t> &tiles) {
  const auto &shape = modelShape();
  const size_t executionTiles = shape.tiles, prefillLength = shape.prefill,
               hiddenSize = shape.hidden, vocabulary = shape.vocabulary,
               layers = shape.layers, kvHeads = shape.kvHeads,
               cacheLength = shape.cache, headSize = shape.head;
  reduced.resize(prefillLength * hiddenSize);
  if (tiles.size() != executionTiles)
    throw std::runtime_error("input tile count differs from compiled model");
  const size_t inputBytes = prefillLength * hiddenSize * sizeof(float);
  const size_t outputBytes =
      std::max(prefillLength * hiddenSize, vocabulary) * sizeof(float);
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

Execution::State::~State() noexcept(false) {
  const auto &shape = modelShape();
  const size_t layers = shape.layers, kvHeads = shape.kvHeads,
               cacheLength = shape.cache, headSize = shape.head;
  if (!closed)
    close();
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
  const auto &shape = modelShape();
  const size_t layers = shape.layers, kvHeads = shape.kvHeads,
               cacheLength = shape.cache, headSize = shape.head;
  const size_t codeBytes = layers * kvHeads * cacheLength * headSize;
  for (auto &worker : workers) {
    std::memset(worker.keyCodes, 0, codeBytes);
    std::memset(worker.valueCodes, 0, codeBytes);
    std::memset(worker.keyScales, 127, codeBytes / 32);
    std::memset(worker.valueScales, 127, codeBytes / 32);
  }
}

#include "execution.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <runtime.h>
#include <sched.h>
#include <signal.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/wait.h>
#include <system_error>
#include <topology.h>
#include <unistd.h>

#include "llama-parameters.h"
constexpr size_t executionTiles = parts;

const ModelShape &modelShape() {
  static const ModelShape shape{hiddenSize,     layers,        kvHeads,
                                hiddenSize,     hiddenSize,    parts,
                                executionTiles, hiddenSize,    headSize,
                                vocabulary,     prefillLength, cacheLength};
  return shape;
}

int controlCpu(size_t tile) {
  for (uint32_t hart = 0; hart < BB_HART_NUM; ++hart) {
    const auto id = bb_topology_core_id(hart);
    if (id.tile == tile && id.core == 0)
      return int(hart);
  }
  throw std::runtime_error("tile has no control CPU");
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

static void runWorker(Worker &worker, size_t rank, int cpu, int64_t *tokens,
                      const std::filesystem::path &directory) {
  cpu_set_t cpus;
  CPU_ZERO(&cpus);
  CPU_SET(cpu, &cpus);
  if (sched_setaffinity(0, sizeof(cpus), &cpus))
    throw std::system_error(errno, std::generic_category(), "pin tile worker");
  runtime_init(1024 * 1024);
  void *workspace = aligned_alloc(64, workspaceBytes);
  if (!workspace)
    throw std::bad_alloc();
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
  for (;;) {
    uint8_t notification;
    if (read(worker.notify[0], &notification, 1) != 1)
      throw std::runtime_error("model command channel closed");
    if (notification == 1)
      break;
    if (notification != 0 || command.state.load(std::memory_order_acquire) != 1)
      throw std::runtime_error("invalid model command notification");
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
    command.state.store(2, std::memory_order_release);
    uint8_t acknowledgement = 0;
    if (write(worker.completion[1], &acknowledgement, 1) != 1)
      throw std::runtime_error("model acknowledgement channel closed");
  }
  free(workspace);
}

Execution::Execution(const std::filesystem::path &directory,
                     const std::vector<uint64_t> &tiles) {
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
  static_assert(std::atomic<int>::is_always_lock_free);
  signal(SIGPIPE, SIG_IGN);
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
    if (pipe(worker.notify) || pipe(worker.completion))
      throw std::system_error(errno, std::generic_category(),
                              "model notification pipe");
  }
  resetCache();
  for (size_t rank = 0; rank < executionTiles; ++rank) {
    auto &worker = workers[rank];
    worker.pid = fork();
    if (worker.pid < 0)
      throw std::system_error(errno, std::generic_category(),
                              "fork tile worker");
    if (worker.pid == 0) {
      for (size_t other = 0; other < executionTiles; ++other) {
        ::close(workers[other].notify[1]);
        ::close(workers[other].completion[0]);
        if (other != rank) {
          ::close(workers[other].notify[0]);
          ::close(workers[other].completion[1]);
        }
      }
      try {
        runWorker(worker, rank, controlCpu(tiles[rank]), tokens, directory);
        std::exit(0);
      } catch (const std::exception &error) {
        std::snprintf(worker.command->error, sizeof(worker.command->error),
                      "%s", error.what());
        worker.command->state.store(4, std::memory_order_release);
        const uint8_t acknowledgement = 1;
        write(worker.completion[1], &acknowledgement, 1);
        std::exit(1);
      }
    }
  }
  for (auto &worker : workers) {
    ::close(worker.notify[0]);
    ::close(worker.completion[1]);
  }
}
Execution::~Execution() noexcept(false) {
  if (!closed)
    close();
  const size_t codeBytes = layers * kvHeads * cacheLength * headSize;
  for (auto &worker : workers) {
    munmap(worker.keyCodes, codeBytes);
    munmap(worker.valueCodes, codeBytes);
    munmap(worker.keyScales, codeBytes / 32);
    munmap(worker.valueScales, codeBytes / 32);
  }
  munmap(storage, storageBytes);
}
void Execution::close() {
  if (closed)
    throw std::runtime_error("model workers already stopped");
  std::string error;
  const uint8_t stop = 1;
  for (auto &worker : workers) {
    if (write(worker.notify[1], &stop, 1) != 1 && error.empty())
      error = "tile worker command channel closed";
    ::close(worker.notify[1]);
  }
  for (auto &worker : workers) {
    int status;
    if (waitpid(worker.pid, &status, 0) != worker.pid) {
      if (error.empty())
        error = "cannot join tile worker";
    } else if ((!WIFEXITED(status) || WEXITSTATUS(status) != 0) &&
               error.empty()) {
      error = "tile worker exited abnormally";
    }
    ::close(worker.completion[0]);
  }
  closed = true;
  if (!error.empty())
    throw std::runtime_error(error);
}
void Execution::submit(size_t rank, uint64_t operation, size_t count,
                       size_t start, size_t layer) {
  auto &worker = workers.at(rank);
  auto &command = *worker.command;
  if (command.state.load(std::memory_order_acquire) != 0)
    throw std::runtime_error("worker is not idle");
  command.operation = operation;
  command.count = count;
  command.start = start;
  command.layer = layer;
  command.state.store(1, std::memory_order_release);
  const uint8_t run = 0;
  if (write(worker.notify[1], &run, 1) != 1)
    throw std::runtime_error("tile worker command channel closed");
}
void Execution::wait(size_t rank) {
  auto &worker = workers.at(rank);
  uint8_t acknowledgement;
  if (read(worker.completion[0], &acknowledgement, 1) != 1)
    throw std::runtime_error("tile worker exited before completion");
  auto &command = *worker.command;
  int state = command.state.load(std::memory_order_acquire);
  if (acknowledgement == 1 && state == 4)
    throw std::runtime_error(command.error);
  if (acknowledgement != 0 || state != 2)
    throw std::runtime_error("invalid tile completion");
  command.state.store(0, std::memory_order_release);
}

void Execution::resetCache() {
  const size_t codeBytes = layers * kvHeads * cacheLength * headSize;
  for (auto &worker : workers) {
    std::memset(worker.keyCodes, 0, codeBytes);
    std::memset(worker.valueCodes, 0, codeBytes);
    std::memset(worker.keyScales, 127, codeBytes / 32);
    std::memset(worker.valueScales, 127, codeBytes / 32);
  }
}

//===- generation.cpp ----------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "execution.h"
#include "gemma-plan.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <signal.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <system_error>
#include <topology.h>
#include <unistd.h>

extern "C" double _mlir_ciface_rtclock() {
  timeval time;
  if (gettimeofday(&time, nullptr))
    throw std::runtime_error("gettimeofday failed");
  return time.tv_sec + time.tv_usec * 1.0e-6;
}

int controlCpu(size_t tile) {
  for (uint32_t hart = 0; hart < BB_HART_NUM; ++hart) {
    auto id = bb_topology_core_id(hart);
    if (id.tile == tile && id.core == 0)
      return int(hart);
  }
  throw std::runtime_error("Gemma tile has no controller CPU");
}

static void runWorker(Execution::Shared &shared,
                      const std::vector<size_t> &tiles, int notify,
                      int completion, const std::filesystem::path &directory) {
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  CPU_SET(controlCpu(0), &affinity);
  if (sched_setaffinity(0, sizeof(affinity), &affinity))
    throw std::system_error(errno, std::generic_category(),
                            "pin Gemma coordinator");
  std::vector<int> cpus;
  for (size_t tile : tiles)
    cpus.push_back(controlCpu(tile));
  auto execution = createGemmaExecution(directory, cpus);
  bool prefilled = false;
  size_t position = 0;
  for (;;) {
    uint8_t operation;
    if (read(notify, &operation, 1) != 1)
      throw std::runtime_error("Gemma command channel closed");
    if (operation == 2)
      break;
    if (operation > 1 || shared.state.load(std::memory_order_acquire) != 1 ||
        (operation == 1 && !prefilled))
      throw std::runtime_error("invalid Gemma command");
    if (operation == 0) {
      if (prefilled)
        throw std::runtime_error("Gemma prefill submitted twice");
      int64_t tokens[PrefillLength] = {}, positions[PrefillLength];
      std::copy_n(shared.tokens, shared.count, tokens);
      for (size_t index = 0; index < PrefillLength; ++index)
        positions[index] = index;
      int64_t valid = shared.count;
      execution->set("prefill.tokens", tokens, sizeof(tokens));
      execution->set("prefill.positions", positions, sizeof(positions));
      execution->set("valid_tokens", &valid, sizeof(valid));
      runPrefill(*execution);
      position = shared.count;
      prefilled = true;
    } else {
      if (position >= CacheLength)
        throw std::runtime_error(
            "Gemma decode exceeds compiled cache capacity");
      int64_t current = position, valid = 1;
      execution->set("decode.tokens", &shared.token, sizeof(shared.token));
      execution->set("decode.positions", &current, sizeof(current));
      execution->set("valid_tokens", &valid, sizeof(valid));
      runDecode(*execution);
      ++position;
    }
    auto logits =
        execution->read(operation == 0 ? "prefill.logits" : "decode.logits");
    std::copy_n(static_cast<float *>(logits.data) + logits.offset, MaxVocabSize,
                shared.logits);
    execution->reset();
    shared.state.store(2, std::memory_order_release);
    uint8_t acknowledgement = 0;
    if (write(completion, &acknowledgement, 1) != 1)
      throw std::runtime_error("Gemma completion channel closed");
  }
}

Execution::Execution(const std::filesystem::path &directory,
                     const std::vector<size_t> &tiles) {
  shared = static_cast<Shared *>(mmap(nullptr, sizeof(Shared),
                                      PROT_READ | PROT_WRITE,
                                      MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  if (shared == MAP_FAILED)
    throw std::system_error(errno, std::generic_category(), "Gemma shared DDR");
  new (shared) Shared{};
  signal(SIGPIPE, SIG_IGN);
  if (pipe(notify) || pipe(completion))
    throw std::system_error(errno, std::generic_category(), "Gemma pipes");
  pid = fork();
  if (pid < 0)
    throw std::system_error(errno, std::generic_category(),
                            "Gemma worker fork");
  if (pid == 0) {
    ::close(notify[1]);
    ::close(completion[0]);
    try {
      runWorker(*shared, tiles, notify[0], completion[1], directory);
      std::exit(0);
    } catch (const std::exception &error) {
      std::snprintf(shared->error, sizeof(shared->error), "%s", error.what());
      shared->state.store(4, std::memory_order_release);
      uint8_t acknowledgement = 1;
      if (write(completion[1], &acknowledgement, 1) != 1)
        std::abort();
      std::exit(1);
    }
  }
  ::close(notify[0]);
  ::close(completion[1]);
}
Execution::~Execution() noexcept(false) {
  if (!closed)
    close();
  munmap(shared, sizeof(Shared));
}
void Execution::run(uint8_t operation) {
  if (shared->state.load(std::memory_order_acquire) != 0)
    throw std::runtime_error("Gemma worker is not idle");
  shared->state.store(1, std::memory_order_release);
  if (write(notify[1], &operation, 1) != 1)
    throw std::runtime_error("Gemma command channel closed");
  uint8_t acknowledgement;
  if (read(completion[0], &acknowledgement, 1) != 1)
    throw std::runtime_error("Gemma worker exited before completion");
  int state = shared->state.load(std::memory_order_acquire);
  if (acknowledgement == 1 && state == 4)
    throw std::runtime_error(shared->error);
  if (acknowledgement != 0 || state != 2)
    throw std::runtime_error("invalid Gemma completion");
  shared->state.store(0, std::memory_order_release);
}
void Execution::prefill(const std::vector<int64_t> &tokens) {
  if (tokens.empty() || tokens.size() > PrefillLength)
    throw std::runtime_error("Gemma input exceeds the compiled prefill length");
  shared->count = tokens.size();
  std::copy(tokens.begin(), tokens.end(), shared->tokens);
  run(0);
}
void Execution::decode(int64_t token) {
  shared->token = token;
  run(1);
}
void Execution::close() {
  if (closed)
    throw std::runtime_error("Gemma worker already closed");
  uint8_t stop = 2;
  std::string error;
  if (write(notify[1], &stop, 1) != 1)
    error = "Gemma command channel closed";
  ::close(notify[1]);
  int status;
  if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status) ||
      WEXITSTATUS(status))
    error = "Gemma worker failed";
  ::close(completion[0]);
  closed = true;
  if (!error.empty())
    throw std::runtime_error(error);
}

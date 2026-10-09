#include "model.h"
#include "stages.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <runtime.h>
#include <sched.h>
#include <signal.h>
#include <sys/wait.h>
#include <topology.h>

const Shape &modelShape() {
  static const Shape shape{length, width, options, actions};
  return shape;
}
int controlCpu(size_t tile) {
  for (uint32_t hart = 0; hart < BB_HART_NUM; ++hart) {
    auto id = bb_topology_core_id(hart);
    if (id.tile == tile && id.core == 0)
      return hart;
  }
  throw std::runtime_error("tile has no control CPU");
}
size_t resourceBytes(const std::filesystem::path &path) {
  return resources::indexed
             ? resources::index.entries
                   .at(path.lexically_normal()
                           .lexically_relative(resources::index.root)
                           .generic_string())
                   .size
             : std::filesystem::file_size(path);
}
struct Command {
  std::atomic<int> state;
  size_t entry;
  char error[256];
};
static void worker(Context &ctx, Command &command, int notify, int completion,
                   const std::filesystem::path &directory, size_t tile) {
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  CPU_SET(controlCpu(tile), &affinity);
  if (sched_setaffinity(0, sizeof(affinity), &affinity))
    throw std::system_error(errno, std::generic_category(), "pin Laya worker");
  runtime_init(1024 * 1024);
  std::vector<std::unique_ptr<Parameters>> parameters;
  for (const auto &entry : entries)
    parameters.push_back(std::make_unique<Parameters>(directory / entry.name));
  constexpr size_t workspaceBytes = size_t(256) << 20;
  void *workspace = aligned_alloc(64, workspaceBytes);
  if (!workspace)
    throw std::bad_alloc();
  workspace_init(workspace, workspaceBytes);
  for (;;) {
    uint8_t message;
    if (read(notify, &message, 1) != 1)
      throw std::runtime_error("Laya command channel closed");
    if (message == 1)
      break;
    if (message || command.state.load(std::memory_order_acquire) != 1)
      throw std::runtime_error("invalid Laya command");
    const auto &entry = entries[command.entry];
    std::string kind = entry.kind;
    workspace_begin(workspace, workspaceBytes);
    entry.run(ctx, *parameters[command.entry]);
    if (kind == "scorer") {
      std::copy_n(ctx.scores.getData(), options, ctx.scoreValues.getData());
      workspace_free(ctx.scores.release());
    } else if (kind == "action") {
      std::copy_n(ctx.action.getData(), actions, ctx.actionValues.getData());
      workspace_free(ctx.action.release());
    } else {
      std::copy_n(ctx.next.getData(), length * width, ctx.hidden.getData());
      workspace_free(ctx.next.release());
    }
    command.state.store(2, std::memory_order_release);
    message = 0;
    if (write(completion, &message, 1) != 1)
      throw std::runtime_error("Laya acknowledgement channel closed");
  }
  free(workspace);
}
void execute(Context &ctx, const std::filesystem::path &directory,
             size_t tile) {
  void *storage = mmap(nullptr, sizeof(Command), PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (storage == MAP_FAILED)
    throw std::system_error(errno, std::generic_category(),
                            "Laya command storage");
  auto &command = *new (storage) Command{};
  int notify[2], completion[2];
  if (pipe(notify) || pipe(completion))
    throw std::system_error(errno, std::generic_category(),
                            "Laya notification pipe");
  signal(SIGPIPE, SIG_IGN);
  pid_t child = fork();
  if (child < 0)
    throw std::system_error(errno, std::generic_category(), "fork Laya worker");
  if (child == 0) {
    close(notify[1]);
    close(completion[0]);
    try {
      worker(ctx, command, notify[0], completion[1], directory, tile);
      std::exit(0);
    } catch (const std::exception &error) {
      std::snprintf(command.error, sizeof(command.error), "%s", error.what());
      command.state.store(4, std::memory_order_release);
      uint8_t message = 1;
      write(completion[1], &message, 1);
      std::exit(1);
    }
  }
  close(notify[0]);
  close(completion[1]);
  std::exception_ptr failure;
  try {
    for (size_t index = 0; index < std::size(entries); ++index) {
      std::string kind = entries[index].kind;
      if (kind == "scorer") {
        for (size_t option = 0; option < options; ++option) {
          auto position = ctx.positions.getData()[option];
          if (position < 0 || size_t(position) >= length)
            throw std::runtime_error("invalid marker position");
          std::copy_n(ctx.hidden.getData() + position * width, width,
                      ctx.markers.getData() + option * width);
        }
      } else if (kind == "action") {
        float maximum = -INFINITY;
        size_t count = 0;
        for (size_t option = 0; option < options; ++option) {
          if (!ctx.valid.getData()[option])
            ctx.scoreValues.getData()[option] = -1e4f;
          else
            ++count;
          maximum = std::max(maximum, ctx.scoreValues.getData()[option]);
        }
        if (!count)
          throw std::runtime_error("Laya requires at least one option");
        count = std::max(count, size_t(2));
        std::vector<float> probabilities(options);
        float sum = 0;
        for (size_t option = 0; option < options; ++option)
          sum += probabilities[option] =
              std::exp(ctx.scoreValues.getData()[option] - maximum);
        float entropy = 0, first = 0, second = 0;
        for (float probability : probabilities) {
          probability /= sum;
          entropy -= probability * std::log(std::max(probability, 1e-9f));
          if (probability > first) {
            second = first;
            first = probability;
          } else
            second = std::max(second, probability);
        }
        std::copy_n(ctx.hidden.getData(), width, ctx.features.getData());
        ctx.features.getData()[width] = first;
        ctx.features.getData()[width + 1] = first - second;
        ctx.features.getData()[width + 2] = entropy / std::log(float(count));
        ctx.features.getData()[width + 3] = float(count) / 255;
      }
      command.entry = index;
      command.state.store(1, std::memory_order_release);
      uint8_t message = 0;
      if (write(notify[1], &message, 1) != 1 ||
          read(completion[0], &message, 1) != 1)
        throw std::runtime_error("Laya worker exited before completion");
      int state = command.state.load(std::memory_order_acquire);
      if (message == 1 && state == 4)
        throw std::runtime_error(command.error);
      if (message || state != 2)
        throw std::runtime_error("invalid Laya completion");
    }
  } catch (...) {
    failure = std::current_exception();
  }
  uint8_t stop = 1;
  int sent = write(notify[1], &stop, 1);
  close(notify[1]);
  int status;
  pid_t joined = waitpid(child, &status, 0);
  close(completion[0]);
  munmap(storage, sizeof(Command));
  if (failure)
    std::rethrow_exception(failure);
  if (sent != 1 || joined != child || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0)
    throw std::runtime_error("Laya worker exited abnormally");
}

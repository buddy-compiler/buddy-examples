#include <algorithm>
#include <buddy/Core/Container.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <runtime.h>
#include <string>
#include <vector>

using Hidden = MemRef<float, 3>;
using Matrix = MemRef<float, 2>;
using Floats = MemRef<float, 1>;
using Bytes = MemRef<int8_t, 1>;
using Tokens = MemRef<int64_t, 2>;
using Index = MemRef<int64_t, 1>;

template <typename T> MemRef<T, 1> load(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  file.exceptions(std::ios::failbit | std::ios::badbit);
  size_t bytes = file.tellg();
  if (bytes % sizeof(T))
    throw std::runtime_error("invalid parameter byte count");
  MemRef<T, 1> value(std::vector<size_t>{bytes / sizeof(T)}, bytes != 0, 0);
  file.seekg(0);
  if (bytes)
    file.read(reinterpret_cast<char *>(value.getData()), bytes);
  return value;
}
struct Parameters {
  Floats floats;
  Bytes bytes;
  explicit Parameters(const std::filesystem::path &path)
      : floats(load<float>(path / "params.f32")),
        bytes(load<int8_t>(path / "weights.bin")) {}
};
struct Context {
  Tokens tokens, mask, positions, valid;
  Index qtype;
  Hidden hidden, next, markers, scores;
  Matrix features, action;
  std::vector<float> scoreValues;
  Context(size_t length, size_t width, size_t options, size_t actions)
      : tokens({1, length}), mask({1, length}), positions({1, options}),
        valid({1, options}), qtype({1}), hidden({1, length, width}),
        next({1, length, width}, false, 0), markers({1, options, width}),
        scores({1, options, 1}, false, 0), features({1, width + 4}),
        action({1, actions}, false, 0), scoreValues(options) {}
};
struct Entry {
  const char *name;
  const char *kind;
  void (*run)(Context &, Parameters &);
};
#include "stages.h"

int main(int argc, char **argv) {
  if (argc != 3)
    throw std::runtime_error("usage: laya-run MODEL_DIRECTORY TRACE_DIRECTORY");
  const std::filesystem::path directory = argv[1];
  const std::filesystem::path traceDirectory = argv[2];
  runtime_init(1024 * 1024);
  std::vector<std::unique_ptr<Parameters>> parameters;
  for (const auto &entry : entries)
    parameters.push_back(std::make_unique<Parameters>(directory / entry.name));
  constexpr size_t workspaceBytes = size_t(256) << 20;
  void *workspace = aligned_alloc(64, workspaceBytes);
  if (!workspace)
    throw std::bad_alloc();
  Context ctx(length, width, options, actions);
  std::cout.exceptions(std::ios::badbit | std::ios::failbit);
  size_t request = 0;
  for (;;) {
    std::cin.read(reinterpret_cast<char *>(ctx.tokens.getData()),
                  length * sizeof(int64_t));
    if (std::cin.eof() && std::cin.gcount() == 0)
      break;
    if (!std::cin)
      throw std::runtime_error("incomplete Laya tokens");
    for (auto *value : {&ctx.mask, &ctx.positions, &ctx.valid})
      std::cin.read(reinterpret_cast<char *>(value->getData()),
                    value->getSize() * sizeof(int64_t));
    std::cin.read(reinterpret_cast<char *>(ctx.qtype.getData()),
                  sizeof(int64_t));
    if (!std::cin)
      throw std::runtime_error("incomplete Laya request");
    if (ctx.qtype[0] < 0 || ctx.qtype[0] > 2)
      throw std::runtime_error("invalid question type");
    size_t peak = 0;
    for (size_t i = 0; i < std::size(entries); ++i) {
      workspace_begin(workspace, workspaceBytes);
      const auto &entry = entries[i];
      std::string kind = entry.kind;
      if (kind == "scorer") {
        for (size_t j = 0; j < options; ++j) {
          auto pos = ctx.positions.getData()[j];
          if (pos < 0 || size_t(pos) >= length)
            throw std::runtime_error("invalid marker position");
          std::copy_n(ctx.hidden.getData() + pos * width, width,
                      ctx.markers.getData() + j * width);
        }
      } else if (kind == "action") {
        float maximum = -INFINITY;
        size_t count = 0;
        for (size_t j = 0; j < options; ++j) {
          if (!ctx.valid.getData()[j])
            ctx.scoreValues[j] = -1e4f;
          else
            ++count;
          maximum = std::max(maximum, ctx.scoreValues[j]);
        }
        if (count == 0)
          throw std::runtime_error("Laya requires at least one option");
        count = std::max(count, size_t(2));
        std::vector<float> probabilities(options);
        float sum = 0;
        for (size_t j = 0; j < options; ++j)
          sum += probabilities[j] = std::exp(ctx.scoreValues[j] - maximum);
        float entropy = 0, first = 0, second = 0;
        for (float p : probabilities) {
          p /= sum;
          entropy -= p * std::log(std::max(p, 1e-9f));
          if (p > first) {
            second = first;
            first = p;
          } else
            second = std::max(second, p);
        }
        std::copy_n(ctx.hidden.getData(), width, ctx.features.getData());
        ctx.features.getData()[width] = first;
        ctx.features.getData()[width + 1] = first - second;
        ctx.features.getData()[width + 2] = entropy / std::log(float(count));
        ctx.features.getData()[width + 3] = float(count) / 255;
      }
#ifdef CTRACE
      uint64_t start, end;
      asm volatile("rdcycle %0" : "=r"(start));
#endif
      entry.run(ctx, *parameters[i]);
#ifdef CTRACE
      asm volatile("rdcycle %0" : "=r"(end));
      std::cerr << entry.name << " controller_cycles=" << end - start
                << " workspace_bytes=" << workspace_peak() << '\n';
#endif
#ifdef DTRACE
      const float *data = kind == "action"   ? ctx.action.getData()
                          : kind == "scorer" ? ctx.scores.getData()
                                             : ctx.next.getData();
      size_t size = kind == "action"   ? actions
                    : kind == "scorer" ? options
                                       : length * width;
      std::ofstream output(traceDirectory / (std::to_string(request) + "-" +
                                             entry.name + ".f32"),
                           std::ios::binary);
      output.exceptions(std::ios::failbit | std::ios::badbit);
      output.write(reinterpret_cast<const char *>(data), size * sizeof(float));
#endif
      peak = std::max(peak, workspace_peak());
      if (kind == "scorer") {
        std::copy_n(ctx.scores.getData(), options, ctx.scoreValues.data());
        workspace_free(ctx.scores.release());
      } else if (kind != "action") {
        std::copy_n(ctx.next.getData(), length * width, ctx.hidden.getData());
        workspace_free(ctx.next.release());
      }
    }
    std::cout.write(reinterpret_cast<const char *>(ctx.scoreValues.data()),
                    options * sizeof(float));
    std::cout.write(reinterpret_cast<const char *>(ctx.action.getData()),
                    actions * sizeof(float));
    std::cout.flush();
    workspace_free(ctx.action.release());
    std::cerr << "request=" << request++ << " workspace_peak_bytes=" << peak
              << '\n';
  }
  free(workspace);
}

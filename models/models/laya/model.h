#pragma once
#include <algorithm>
#include <buddy/Core/Container.h>
#include <filesystem>
#include <resources.h>
#include <system_error>
#include <vector>
using Hidden = MemRef<float, 3>;
using Matrix = MemRef<float, 2>;
using Floats = MemRef<float, 1>;
using Bytes = MemRef<int8_t, 1>;
using Tokens = MemRef<int64_t, 2>;
using Index = MemRef<int64_t, 1>;
size_t resourceBytes(const std::filesystem::path &path);
template <typename T> class Parameter : public MemRef<T, 1> {
  resources::Mapping mapping;

public:
  explicit Parameter(const std::filesystem::path &path)
      : MemRef<T, 1>(std::vector<size_t>{resourceBytes(path) / sizeof(T)},
                     false, 0),
        mapping(path, resourceBytes(path), alignof(T)) {
    if (resourceBytes(path) % sizeof(T))
      throw std::runtime_error("invalid parameter byte count");
    this->aligned = static_cast<T *>(mapping.data());
  }
};
struct Parameters {
  Parameter<float> floats;
  Parameter<int8_t> bytes;
  explicit Parameters(const std::filesystem::path &path)
      : floats(path / "params.f32"), bytes(path / "weights.bin") {}
};
template <typename T, size_t N> class Shared : public MemRef<T, N> {
public:
  explicit Shared(std::vector<size_t> sizes) : MemRef<T, N>(sizes, false, 0) {
    void *data =
        mmap(nullptr, this->getSize() * sizeof(T), PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED)
      throw std::system_error(errno, std::generic_category(),
                              "shared Laya tensor");
    this->aligned = static_cast<T *>(data);
  }
  ~Shared() { munmap(this->aligned, this->getSize() * sizeof(T)); }
};
struct Context {
  Shared<int64_t, 2> tokens, mask, positions, valid;
  Shared<int64_t, 1> qtype;
  Shared<float, 3> hidden, markers;
  Hidden next, scores;
  Shared<float, 2> features;
  Matrix action;
  Shared<float, 1> scoreValues, actionValues;
  Context(size_t length, size_t width, size_t options, size_t actions)
      : tokens({1, length}), mask({1, length}), positions({1, options}),
        valid({1, options}), qtype({1}), hidden({1, length, width}),
        markers({1, options, width}), next({1, length, width}, false, 0),
        scores({1, options, 1}, false, 0), features({1, width + 4}),
        action({1, actions}, false, 0), scoreValues({options}),
        actionValues({actions}) {}
};
struct Entry {
  const char *name;
  const char *kind;
  void (*run)(Context &, Parameters &);
};
void execute(Context &, const std::filesystem::path &directory, size_t tile);
void decision(Context &, size_t options, size_t actions, double temperature);

struct Shape {
  size_t length, width, options, actions;
};
const Shape &modelShape();
int controlCpu(size_t tile);

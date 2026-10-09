#pragma once

#include "task_worker.h"
#include <initializer_list>
#include <map>
#include <memory>
#include <resources.h>
#include <string>
#include <vector>

struct InputBinding {
  const char *slot;
  const char *resource;
  size_t byteOffset;
  BufferView layout;
};
struct ResourceSpec {
  const char *name;
  size_t bytes, alignment;
};
struct SlotSpec {
  const char *name;
  BufferView layout;
  size_t rank, elementBytes;
  bool reserveStorage;
};
struct ColumnSpan {
  size_t start, count;
};

class LayerExecution {
  struct Slot {
    BufferView storage, value;
    size_t rank, elementBytes;
  };
  struct Pending {
    std::vector<std::string> outputs;
    std::vector<bool> owned;
  };
  std::map<std::string, std::unique_ptr<resources::Mapping>> mappings;
  std::map<std::string, Slot> slots;
  std::vector<std::unique_ptr<TaskWorker>> workers;
  std::vector<Pending> pending;
  std::map<std::string, size_t> producing;
  std::map<void *, size_t> allocations;
  void *memory;
  size_t memoryBytes;
  void wait(size_t worker);

public:
  LayerExecution(const std::filesystem::path &directory,
                 const std::vector<int> &cpus,
                 std::initializer_list<ResourceSpec> resources,
                 std::initializer_list<SlotSpec> slots, size_t workspaceBytes);
  ~LayerExecution() noexcept(false);
  void compute(int rank, size_t kernel,
               std::initializer_list<InputBinding> inputs,
               std::initializer_list<const char *> outputs,
               std::initializer_list<bool> owned);
  BufferView read(const std::string &slot);
  void set(const std::string &slot, const void *data, size_t bytes);
  void gather(std::initializer_list<const char *> inputs,
              std::initializer_list<ColumnSpan> columns,
              const std::string &output);
  void view(const std::string &input, const std::string &output, size_t axis,
            int64_t start, int64_t count);
  void lastValidRow(const std::string &input, const std::string &output,
                    const std::string &count);
  void cacheUpdate(std::initializer_list<const char *> inputs,
                   std::initializer_list<const char *> outputs,
                   const std::string &length, const std::string &count);
  void reset();
  void keepLive(std::initializer_list<const char *> names);
};

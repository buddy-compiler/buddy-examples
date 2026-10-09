#include "layer_execution.h"
#include <cerrno>
#include <cstring>
#include <runtime.h>
#include <set>
#include <stdexcept>
#include <sys/mman.h>
#include <system_error>

LayerExecution::LayerExecution(
    const std::filesystem::path &directory, const std::vector<int> &cpus,
    std::initializer_list<ResourceSpec> resourceSpecs,
    std::initializer_list<SlotSpec> slotSpecs, size_t workspaceBytes) {
  for (const auto &resource : resourceSpecs)
    mappings.emplace(resource.name, std::make_unique<resources::Mapping>(
                                        directory / resource.name,
                                        resource.bytes, resource.alignment));
  memoryBytes = cpus.size() * workspaceBytes;
  for (const auto &slot : slotSpecs) {
    if (!slot.reserveStorage)
      continue;
    size_t bytes = slot.elementBytes;
    for (size_t axis = 0; axis < slot.rank; ++axis)
      bytes *= slot.layout.sizes[axis];
    memoryBytes += (bytes + 15) & ~size_t(15);
  }
  memory = mmap(nullptr, memoryBytes, PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (memory == MAP_FAILED)
    throw std::system_error(errno, std::generic_category(),
                            "allocate layer storage");
  workspace_init(memory, memoryBytes);
  auto *cursor = static_cast<unsigned char *>(memory);
  for (const auto &spec : slotSpecs) {
    auto value = spec.layout;
    value.allocated = value.data = spec.reserveStorage ? cursor : nullptr;
    if (!slots
             .emplace(spec.name,
                      Slot{value, value, spec.rank, spec.elementBytes})
             .second)
      throw std::runtime_error("duplicate execution slot");
    size_t bytes = spec.elementBytes;
    for (size_t axis = 0; axis < spec.rank; ++axis)
      bytes *= value.sizes[axis];
    if (spec.reserveStorage)
      cursor += (bytes + 15) & ~size_t(15);
  }
  // Every worker inherits all input buffers and all worker arenas before fork.
  for (int cpu : cpus) {
    workers.push_back(
        std::make_unique<TaskWorker>(cpu, cursor, workspaceBytes));
    cursor += workspaceBytes;
  }
  pending.resize(workers.size());
}

void LayerExecution::wait(size_t worker) {
  auto &job = pending.at(worker);
  if (job.outputs.empty())
    return;
  BufferView outputs[GemmaTaskMaxOutputs];
  workers.at(worker)->wait(outputs, job.outputs.size());
  for (size_t index = 0; index < job.outputs.size(); ++index) {
    slots.at(job.outputs[index]).value = outputs[index];
    producing.erase(job.outputs[index]);
    if (job.owned[index])
      allocations.emplace(outputs[index].allocated, worker);
  }
  job = {};
}

BufferView LayerExecution::read(const std::string &name) {
  auto producer = producing.find(name);
  if (producer != producing.end())
    wait(producer->second);
  return slots.at(name).value;
}

void LayerExecution::compute(int rank, size_t kernel,
                             std::initializer_list<InputBinding> bindings,
                             std::initializer_list<const char *> outputNames,
                             std::initializer_list<bool> owned) {
  size_t worker = rank < 0 ? 0 : size_t(rank);
  wait(worker);
  if (owned.size() != outputNames.size())
    throw std::runtime_error("task output ownership count mismatch");
  std::vector<BufferView> inputs;
  for (const auto &binding : bindings) {
    if (binding.slot) {
      inputs.push_back(read(binding.slot));
    } else {
      auto value = binding.layout;
      value.allocated = mappings.at(binding.resource)->data();
      value.data =
          static_cast<unsigned char *>(value.allocated) + binding.byteOffset;
      inputs.push_back(value);
    }
  }
  auto &job = pending.at(worker);
  job.outputs.assign(outputNames.begin(), outputNames.end());
  job.owned.assign(owned.begin(), owned.end());
  for (const auto &name : job.outputs)
    if (!producing.emplace(name, worker).second)
      throw std::runtime_error("execution slot already has an active producer");
  workers.at(worker)->submit(kernel, inputs.data(), inputs.size());
}

void LayerExecution::set(const std::string &name, const void *data,
                         size_t bytes) {
  auto &slot = slots.at(name);
  size_t expected = slot.elementBytes;
  for (size_t axis = 0; axis < slot.rank; ++axis)
    expected *= slot.storage.sizes[axis];
  if (!slot.storage.data || bytes != expected || producing.count(name))
    throw std::runtime_error("execution input size or lifetime mismatch");
  std::memcpy(slot.storage.data, data, bytes);
  slot.value = slot.storage;
}

void LayerExecution::gather(std::initializer_list<const char *> inputs,
                            std::initializer_list<ColumnSpan> columns,
                            const std::string &outputName) {
  if (inputs.size() != columns.size())
    throw std::runtime_error("gather source count mismatch");
  auto &target = slots.at(outputName);
  if (target.rank != 3)
    throw std::runtime_error("column gather requires a rank-three tensor");
  target.value = target.storage;
  auto column = columns.begin();
  for (const char *name : inputs) {
    auto source = read(name);
    if (slots.at(name).elementBytes != target.elementBytes ||
        source.strides[2] != 1 || source.sizes[0] != target.value.sizes[0] ||
        source.sizes[1] != target.value.sizes[1] ||
        size_t(source.sizes[2]) != column->count ||
        column->start + column->count > size_t(target.value.sizes[2]))
      throw std::runtime_error("gather source shape mismatch");
    for (int64_t batch = 0; batch < source.sizes[0]; ++batch)
      for (int64_t row = 0; row < source.sizes[1]; ++row) {
        size_t from =
            source.offset + batch * source.strides[0] + row * source.strides[1];
        size_t to = batch * target.value.strides[0] +
                    row * target.value.strides[1] + column->start;
        std::memcpy(static_cast<unsigned char *>(target.value.data) +
                        to * target.elementBytes,
                    static_cast<unsigned char *>(source.data) +
                        from * target.elementBytes,
                    column->count * target.elementBytes);
      }
    ++column;
  }
}

void LayerExecution::view(const std::string &input, const std::string &output,
                          size_t axis, int64_t start, int64_t count) {
  auto value = read(input);
  if (axis >= slots.at(input).rank || start < 0 || count <= 0 ||
      start + count > value.sizes[axis])
    throw std::runtime_error("execution view lies outside its source");
  value.offset += start * value.strides[axis];
  value.sizes[axis] = count;
  slots.at(output).value = value;
}

void LayerExecution::lastValidRow(const std::string &input,
                                  const std::string &output,
                                  const std::string &count) {
  auto value = read(count);
  int64_t valid = static_cast<int64_t *>(value.data)[value.offset];
  view(input, output, 1, valid - 1, 1);
}

void LayerExecution::cacheUpdate(std::initializer_list<const char *> inputs,
                                 std::initializer_list<const char *> outputs,
                                 const std::string &length,
                                 const std::string &count) {
  if (inputs.size() != 4 || outputs.size() != 4)
    throw std::runtime_error(
        "MXFP8 KV update requires four code/scale tensors");
  auto position = read(length), valid = read(count);
  int64_t first = static_cast<int64_t *>(position.data)[position.offset];
  int64_t rows = static_cast<int64_t *>(valid.data)[valid.offset];
  auto output = outputs.begin();
  for (const char *name : inputs) {
    auto source = read(name);
    auto &target = slots.at(*output++);
    if (target.elementBytes != 1 || target.rank != 4 || first < 0 ||
        rows <= 0 || rows > source.sizes[2] ||
        first + rows > target.value.sizes[2] || source.strides[3] != 1 ||
        target.value.strides[3] != 1 ||
        source.sizes[0] != target.value.sizes[0] ||
        source.sizes[1] != target.value.sizes[1] ||
        source.sizes[3] != target.value.sizes[3])
      throw std::runtime_error("MXFP8 KV update shape or range mismatch");
    for (int64_t batch = 0; batch < source.sizes[0]; ++batch)
      for (int64_t head = 0; head < source.sizes[1]; ++head)
        for (int64_t row = 0; row < rows; ++row) {
          size_t from = source.offset + batch * source.strides[0] +
                        head * source.strides[1] + row * source.strides[2];
          size_t to = target.value.offset + batch * target.value.strides[0] +
                      head * target.value.strides[1] +
                      (first + row) * target.value.strides[2];
          std::memcpy(static_cast<unsigned char *>(target.value.data) + to,
                      static_cast<unsigned char *>(source.data) + from,
                      source.sizes[3]);
        }
  }
  static_cast<int64_t *>(position.data)[position.offset] = first + rows;
}

void LayerExecution::reset() { keepLive({}); }

void LayerExecution::keepLive(std::initializer_list<const char *> names) {
  for (size_t worker = 0; worker < workers.size(); ++worker)
    wait(worker);
  std::set<void *> live;
  std::set<std::string> liveNames;
  for (const char *name : names) {
    live.insert(slots.at(name).value.allocated);
    liveNames.insert(name);
  }
  for (auto it = allocations.begin(); it != allocations.end();) {
    if (live.count(it->first)) {
      ++it;
    } else {
      workers.at(it->second)->release(it->first);
      it = allocations.erase(it);
    }
  }
  for (auto &[name, slot] : slots)
    if (!liveNames.count(name))
      slot.value = slot.storage;
}

LayerExecution::~LayerExecution() noexcept(false) {
  reset();
  workers.clear();
  munmap(memory, memoryBytes);
}

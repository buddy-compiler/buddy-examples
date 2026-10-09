#pragma once
#include <buddy/Core/Container.h>
#include <resources.h>

template <typename T> class ReadonlyMappedMemRef : public MemRef<T, 1> {
  resources::Mapping mapping;

public:
  ReadonlyMappedMemRef(const std::filesystem::path &path, size_t elements,
                       size_t alignment)
      : MemRef<T, 1>(std::vector<size_t>{elements}, false, 0),
        mapping(path, elements * sizeof(T), alignment) {
    this->aligned = static_cast<T *>(mapping.data());
  }
  ReadonlyMappedMemRef(const ReadonlyMappedMemRef &) = delete;
  ReadonlyMappedMemRef &operator=(const ReadonlyMappedMemRef &) = delete;
};

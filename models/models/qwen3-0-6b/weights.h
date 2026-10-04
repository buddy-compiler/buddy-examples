#pragma once
#include <buddy/Core/Container.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fcntl.h>
#include <stdexcept>
#include <system_error>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct WeightRange { size_t offset, bytes; };

template <typename T> class ReadonlyMappedMemRef : public MemRef<T, 1> {
  size_t mappedBytes;

public:
  ReadonlyMappedMemRef(const std::filesystem::path &path, size_t elements,
                      size_t alignment)
      : MemRef<T, 1>(std::vector<size_t>{elements}, false, 0),
        mappedBytes(elements * sizeof(T)) {
    int descriptor = open(path.c_str(), O_RDONLY);
    if (descriptor < 0)
      throw std::system_error(errno, std::generic_category(), path.string());
    try {
      struct stat info;
      if (fstat(descriptor, &info))
        throw std::system_error(errno, std::generic_category(), path.string());
      if (info.st_size < 0 || uint64_t(info.st_size) != mappedBytes)
        throw std::runtime_error("weight file length differs from metadata: " + path.string());
      if (mappedBytes) {
        size_t pageBytes = sysconf(_SC_PAGESIZE);
        if (!alignment || alignment % alignof(T))
          throw std::runtime_error("invalid weight alignment: " + path.string());
        size_t pageLength = (mappedBytes + pageBytes - 1) / pageBytes * pageBytes;
        size_t reserveLength = pageLength + alignment;
        void *reservation = mmap(nullptr, reserveLength, PROT_NONE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (reservation == MAP_FAILED)
          throw std::system_error(errno, std::generic_category(), path.string());
        uintptr_t base = reinterpret_cast<uintptr_t>(reservation);
        uintptr_t aligned = (base + alignment - 1) / alignment * alignment;
        void *mapping = mmap(reinterpret_cast<void *>(aligned), mappedBytes,
                             PROT_READ, MAP_PRIVATE | MAP_FIXED, descriptor, 0);
        if (mapping == MAP_FAILED) {
          int error = errno;
          munmap(reservation, reserveLength);
          throw std::system_error(error, std::generic_category(), path.string());
        }
        if (mlock(mapping, mappedBytes)) {
          int error = errno;
          munmap(reservation, reserveLength);
          throw std::system_error(error, std::generic_category(), path.string());
        }
        for (size_t offset = 0; offset < mappedBytes; offset += 4096)
          (void)static_cast<volatile const unsigned char *>(mapping)[offset];
        if (aligned > base)
          munmap(reservation, aligned - base);
        uintptr_t end = base + (reserveLength + pageBytes - 1) / pageBytes * pageBytes;
        if (aligned + pageLength < end)
          munmap(reinterpret_cast<void *>(aligned + pageLength), end - aligned - pageLength);
        this->aligned = static_cast<T *>(mapping);
      }
    } catch (...) {
      close(descriptor);
      throw;
    }
    close(descriptor);
  }

  ReadonlyMappedMemRef(const ReadonlyMappedMemRef &) = delete;
  ReadonlyMappedMemRef &operator=(const ReadonlyMappedMemRef &) = delete;

  ~ReadonlyMappedMemRef() {
    if (this->aligned)
      munmap(this->aligned, mappedBytes);
  }
};

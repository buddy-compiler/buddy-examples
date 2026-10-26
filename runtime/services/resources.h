// Files of the model image in the DDR (see models/tools/model-pack.cpp),
// mapped through /dev/mem. The image sits at the model region at the top of
// the DDR; its header indexes the files.
#pragma once
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <system_error>
#include <unistd.h>

namespace resources {
struct Entry {
  uint64_t offset, bytes;
};
// There is no model image where one was looked for.
struct NoImage : std::runtime_error {
  NoImage() : std::runtime_error("no model image at the model region") {}
};
struct Image {
  uint64_t base = 0;
  std::map<std::string, Entry> files;
};
inline Image image;

// Reads the header of the image at physical address `base`.
inline void configure(uint64_t base) {
  const int fd = open("/dev/mem", O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    throw std::system_error(errno, std::generic_category(), "open /dev/mem");
  const size_t page = sysconf(_SC_PAGESIZE);
  // The header is a multiple of a page; its first line says how long the
  // image is, and the index follows. Map pages until the blank line.
  std::string header;
  for (size_t offset = 0;; offset += page) {
    void *address =
        mmap(nullptr, page, PROT_READ, MAP_SHARED, fd, base + offset);
    if (address == MAP_FAILED) {
      const int error = errno;
      close(fd);
      throw std::system_error(error, std::generic_category(),
                              "map model header");
    }
    header.append(static_cast<const char *>(address), page);
    munmap(address, page);
    if (header.find("\n\n") != std::string::npos)
      break;
  }
  close(fd);
  std::istringstream stream(header);
  std::string line, magic;
  std::getline(stream, line);
  std::istringstream first(line);
  uint64_t bytes;
  if (!(first >> magic >> bytes) || magic != "BBMODEL1")
    throw NoImage();
  image.base = base;
  image.files.clear();
  while (std::getline(stream, line) && !line.empty()) {
    std::istringstream fields(line);
    std::string path, offset, size;
    std::getline(fields, path, '\t');
    std::getline(fields, offset, '\t');
    std::getline(fields, size);
    image.files[path] = {std::stoull(offset), std::stoull(size)};
  }
}

// A file of the image, read-only, aligned to `alignment`.
class Mapping {
  void *address = nullptr;
  size_t mapped = 0;

public:
  Mapping(const std::string &path, size_t bytes, size_t alignment) {
    const auto found = image.files.find(path);
    if (found == image.files.end() || found->second.bytes != bytes)
      throw std::runtime_error("model file missing or of another size: " +
                               path);
    if (!bytes)
      return;
    const size_t page = sysconf(_SC_PAGESIZE);
    mapped = (bytes + page - 1) / page * page;
    // Reserve room to align the mapping, then map over the reservation.
    const size_t reserved = mapped + alignment;
    void *reservation =
        mmap(nullptr, reserved, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (reservation == MAP_FAILED)
      throw std::system_error(errno, std::generic_category(),
                              "reserve model file");
    const uintptr_t base = reinterpret_cast<uintptr_t>(reservation);
    const uintptr_t aligned =
        (base + alignment - 1) & ~(uintptr_t(alignment) - 1);
    const int fd = open("/dev/mem", O_RDONLY | O_CLOEXEC);
    address =
        mmap(reinterpret_cast<void *>(aligned), mapped, PROT_READ,
             MAP_FIXED | MAP_SHARED, fd, image.base + found->second.offset);
    const int error = errno;
    close(fd);
    if (address == MAP_FAILED) {
      munmap(reservation, reserved);
      address = nullptr;
      throw std::system_error(error, std::generic_category(),
                              "map model file: " + path);
    }
    if (aligned != base)
      munmap(reservation, aligned - base);
    const uintptr_t end = base + reserved;
    if (aligned + mapped != end)
      munmap(reinterpret_cast<void *>(aligned + mapped),
             end - aligned - mapped);
  }
  Mapping(const Mapping &) = delete;
  Mapping &operator=(const Mapping &) = delete;
  ~Mapping() {
    if (address)
      munmap(address, mapped);
  }
  void *data() const { return address; }
};
} // namespace resources

#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace resources {
struct Entry { uint64_t offset, size; };
struct Index {
  std::filesystem::path root;
  uint64_t base = 0, size = 0;
  std::map<std::string, Entry> entries;
};
inline Index index;
inline bool indexed = false;
inline uint64_t number(const std::string &text) {
  size_t used;
  uint64_t value = std::stoull(text, &used);
  if (used != text.size() || text.empty() || text.front() == '-') throw std::runtime_error("invalid resource index number");
  return value;
}
inline void configure(const std::filesystem::path &root, const char *path) {
  if (indexed) throw std::runtime_error("resource index configured twice");
  if (!path) return;
  std::ifstream stream(path);
  std::string line, magic, base, size, hash;
  if (!std::getline(stream, line)) throw std::runtime_error("cannot read resource index");
  std::istringstream header(line);
  if (!(header >> magic >> base >> size >> hash) || magic != "BBMODEL1" || hash.size() != 64)
    throw std::runtime_error("invalid resource index header");
  index.root = std::filesystem::weakly_canonical(root);
  index.base = number(base); index.size = number(size);
  if (index.base < 0x80000000ULL || index.base % 4096 || index.base >= 0x80000000ULL + (16ULL << 30) || !index.size || index.size % 4096 ||
      index.size > (0x80000000ULL + (16ULL << 30)) - index.base)
    throw std::runtime_error("resource index outside DDR");
  uint64_t end = 0;
  while (std::getline(stream, line)) {
    std::istringstream fields(line);
    std::string name, offset, bytes, sha;
    if (!std::getline(fields, name, '\t') || !std::getline(fields, offset, '\t') ||
        !std::getline(fields, bytes, '\t') || !std::getline(fields, sha) || sha.size() != 64)
      throw std::runtime_error("invalid resource index record");
    std::filesystem::path relative(name);
    for (const auto &part : relative)
      if (part == "..") throw std::runtime_error("resource path escapes root");
    if (relative.empty() || relative.is_absolute()) throw std::runtime_error("invalid resource path");
    Entry entry{number(offset), number(bytes)};
    if (entry.offset % 4096 || entry.offset < end || entry.offset > index.size || entry.size > index.size - entry.offset ||
        !index.entries.emplace(relative.generic_string(), entry).second)
      throw std::runtime_error("overlapping or invalid resource extent");
    end = entry.offset + entry.size;
  }
  if (!stream.eof() || index.entries.empty()) throw std::runtime_error("empty or unreadable resource index");
  indexed = true;
}

class Mapping {
  void *address = nullptr;
  size_t mapped = 0;
public:
  Mapping(const std::filesystem::path &path, size_t bytes, size_t alignment) {
    uint64_t offset = 0;
    int fd;
    if (indexed) {
      auto relative = path.lexically_normal().lexically_relative(index.root).generic_string();
      const auto found = index.entries.find(relative);
      if (found == index.entries.end() || found->second.size != bytes) throw std::runtime_error("resource index length/path mismatch: " + path.string());
      offset = index.base + found->second.offset;
      fd = open("/dev/mem", O_RDONLY | O_CLOEXEC);
    } else {
      fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    }
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "open resource: " + path.string());
    if (!indexed) {
      struct stat info {};
      if (fstat(fd, &info) || info.st_size < 0 || uint64_t(info.st_size) != bytes) {
        close(fd); throw std::runtime_error("resource file length mismatch: " + path.string());
      }
    }
    if (!bytes) { close(fd); return; }
    size_t page = sysconf(_SC_PAGESIZE);
    if (!alignment || (alignment & (alignment - 1)) || page != 4096 || bytes > SIZE_MAX - page - alignment) {
      close(fd); throw std::runtime_error("invalid resource mapping geometry");
    }
    mapped = (bytes + page - 1) / page * page;
    size_t reserved = mapped + alignment;
    void *reservation = mmap(nullptr, reserved, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (reservation == MAP_FAILED) { int error = errno; close(fd); throw std::system_error(error, std::generic_category(), "reserve resource"); }
    uintptr_t base = reinterpret_cast<uintptr_t>(reservation);
    uintptr_t aligned = (base + alignment - 1) & ~(uintptr_t(alignment) - 1);
    address = mmap(reinterpret_cast<void *>(aligned), mapped, PROT_READ,
                   MAP_FIXED | (indexed ? MAP_SHARED : MAP_PRIVATE), fd, offset);
    int error = errno;
    close(fd);
    if (address == MAP_FAILED) {
      munmap(reservation, reserved); address = nullptr;
      throw std::system_error(error, std::generic_category(), "map resource: " + path.string());
    }
    // PFN mappings are already reserved physical storage, not lockable page-cache pages.
    if (!indexed && mlock(address, mapped)) {
      error = errno; munmap(reservation, reserved); address = nullptr;
      throw std::system_error(error, std::generic_category(), "lock resource");
    }
    for (size_t byte = 0; byte < bytes; byte += page)
      (void)static_cast<volatile const unsigned char *>(address)[byte];
    (void)static_cast<volatile const unsigned char *>(address)[bytes - 1];
    if (aligned != base) munmap(reservation, aligned - base);
    uintptr_t end = base + (reserved + page - 1) / page * page;
    if (aligned + mapped != end) munmap(reinterpret_cast<void *>(aligned + mapped), end - aligned - mapped);
  }
  Mapping(const Mapping &) = delete;
  Mapping &operator=(const Mapping &) = delete;
  ~Mapping() { if (address) munmap(address, mapped); }
  void *data() const { return address; }
};
}

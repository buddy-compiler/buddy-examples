#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <fcntl.h>

int main(int argc, char **argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: dax-check MODEL_DIR RESOURCE_LIST");
    std::ifstream list(argv[2]);
    if (!list) throw std::runtime_error("cannot open resource list");
    std::string line;
    size_t count = 0;
    while (std::getline(list, line)) {
      const auto separator = line.find('\t');
      if (separator == std::string::npos) throw std::runtime_error("invalid resource list record");
      const auto relative = std::filesystem::path(line.substr(0, separator));
      if (relative.empty() || relative.is_absolute()) throw std::runtime_error("invalid resource path");
      for (const auto &part : relative)
        if (part == "..") throw std::runtime_error("resource escapes model directory");
      size_t parsed;
      const auto length = std::stoull(line.substr(separator + 1), &parsed);
      if (parsed != line.size() - separator - 1) throw std::runtime_error("invalid resource length");
      const auto path = std::filesystem::path(argv[1]) / relative;
      struct statx info {};
      if (statx(AT_FDCWD, path.c_str(), AT_SYMLINK_NOFOLLOW,
                STATX_TYPE | STATX_SIZE, &info) != 0)
        throw std::runtime_error("statx failed: " + path.string() + " errno=" + std::to_string(errno));
      if ((info.stx_mask & (STATX_TYPE | STATX_SIZE)) != (STATX_TYPE | STATX_SIZE) ||
          !S_ISREG(info.stx_mode) || info.stx_size != length ||
          !(info.stx_attributes_mask & STATX_ATTR_DAX) || !(info.stx_attributes & STATX_ATTR_DAX))
        throw std::runtime_error("resource is not a verified DAX regular file: " + path.string());
      ++count;
    }
    if (!list.eof() || count == 0) throw std::runtime_error("empty or unreadable resource list");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

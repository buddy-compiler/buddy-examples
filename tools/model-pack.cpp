// model-pack <out> <root> [--mapped] <name>...
//
// Packs explicit paths relative to <root> into one image: a header
// that indexes the files, then each file at a 4 KiB boundary. The image is
// loaded at the model region of the DDR; programs find the files through the
// header (runtime/services/resources.h).
//
// Header, padded with zeros to a 4 KiB boundary:
//   BBMODEL2 <image bytes>\n
//   <path relative to root>\t<offset>\t<bytes>\t<file|mapped>\n
//   \n
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;
struct Resource {
  fs::path path;
  bool mapped;
};
constexpr uint64_t page = 4096;
constexpr uint64_t align(uint64_t value) {
  return (value + page - 1) / page * page;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: model-pack <out> <root> [--mapped] <name>...\n";
    return 1;
  }
  const fs::path root = fs::canonical(argv[2]);
  std::vector<Resource> files;
  for (int i = 3; i < argc; ++i) {
    const bool mapped = std::string(argv[i]) == "--mapped";
    if (mapped && ++i == argc)
      throw std::runtime_error("--mapped requires a resource path");
    files.push_back({argv[i], mapped});
  }
  std::sort(
      files.begin(), files.end(),
      [](const Resource &a, const Resource &b) { return a.path < b.path; });

  // The header size depends on the offsets, which depend on the header size:
  // grow it until it holds its own index.
  uint64_t header = page;
  std::string index;
  for (;; header += page) {
    std::ostringstream text;
    uint64_t offset = header;
    for (const auto &file : files) {
      const uint64_t bytes = fs::file_size(root / file.path);
      text << file.path.generic_string() << '\t' << offset << '\t' << bytes
           << '\t' << (file.mapped ? "mapped" : "file") << '\n';
      offset += align(bytes);
    }
    index = "BBMODEL2 " + std::to_string(offset) + "\n" + text.str() + "\n";
    if (index.size() <= header)
      break;
  }

  std::ofstream out(argv[1], std::ios::binary | std::ios::trunc);
  out << index << std::string(header - index.size(), '\0');
  for (const auto &file : files) {
    const uint64_t bytes = fs::file_size(root / file.path);
    if (bytes) {
      std::ifstream in(root / file.path, std::ios::binary);
      out << in.rdbuf();
    }
    out << std::string(align(bytes) - bytes, '\0');
  }
  if (!out)
    throw std::runtime_error("cannot write " + std::string(argv[1]));
  std::cout << "packed " << files.size() << " files into " << argv[1] << '\n';
}

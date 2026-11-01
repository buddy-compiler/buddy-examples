// model-pack <out> <root> <name>...
//
// Packs the files called <name> found under <root> into one image: a header
// that indexes the files, then each file at a 4 KiB boundary. The image is
// loaded at the model region of the DDR; programs find the files through the
// header (runtime/services/resources.h).
//
// Header, padded with zeros to a 4 KiB boundary:
//   BBMODEL1 <image bytes>\n
//   <path relative to root>\t<offset>\t<bytes>\n   per file
//   \n
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;
constexpr uint64_t page = 4096;
constexpr uint64_t align(uint64_t value) {
  return (value + page - 1) / page * page;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: model-pack <out> <root> <name>...\n";
    return 1;
  }
  const fs::path root = fs::canonical(argv[2]);
  const std::set<std::string> names(argv + 3, argv + argc);
  std::vector<fs::path> files;
  for (const auto &entry : fs::recursive_directory_iterator(root)) {
    const auto relative = fs::relative(entry.path(), root);
    // rax-pack leaves a copy of each stage's files under <stage>.payload/.
    if (entry.is_regular_file() &&
        names.contains(relative.filename().string()) &&
        relative.generic_string().find(".payload/") == std::string::npos)
      files.push_back(relative);
  }
  std::sort(files.begin(), files.end());

  // The header size depends on the offsets, which depend on the header size:
  // grow it until it holds its own index.
  uint64_t header = page;
  std::string index;
  for (;; header += page) {
    std::ostringstream text;
    uint64_t offset = header;
    for (const auto &file : files) {
      const uint64_t bytes = fs::file_size(root / file);
      text << file.generic_string() << '\t' << offset << '\t' << bytes << '\n';
      offset += align(bytes);
    }
    index = "BBMODEL1 " + std::to_string(offset) + "\n" + text.str() + "\n";
    if (index.size() <= header)
      break;
  }

  std::ofstream out(argv[1], std::ios::binary | std::ios::trunc);
  out << index << std::string(header - index.size(), '\0');
  for (const auto &file : files) {
    const uint64_t bytes = fs::file_size(root / file);
    if (bytes) {
      std::ifstream in(root / file, std::ios::binary);
      out << in.rdbuf();
    }
    out << std::string(align(bytes) - bytes, '\0');
  }
  if (!out)
    throw std::runtime_error("cannot write " + std::string(argv[1]));
  std::cout << "packed " << files.size() << " files into " << argv[1] << '\n';
}

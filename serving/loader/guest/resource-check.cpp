#include <resources.h>
#include <iostream>
#include <cstdio>
int main(int argc, char **argv) {
  try {
    if (argc != 5) throw std::runtime_error("usage: resource-check INDEX ROOT RESOURCE PREFIX_HEX");
    resources::configure(argv[2], argv[1]);
    const auto &entry = resources::index.entries.at(argv[3]);
    std::string expected(argv[4]);
    if (expected.size() != 32 || entry.size < 16) throw std::runtime_error("invalid resource check prefix");
    resources::Mapping mapping(std::filesystem::path(argv[2]) / argv[3], entry.size, 4096);
    const auto *bytes = static_cast<volatile const unsigned char *>(mapping.data());
    for (size_t i = 0; i < 16; ++i)
      if (bytes[i] != std::stoul(expected.substr(2 * i, 2), nullptr, 16))
        throw std::runtime_error("DDR mapped resource data mismatch");
    std::ifstream smaps("/proc/self/smaps");
    std::string line;
    bool active = false, checked = false;
    while (std::getline(smaps, line)) {
      unsigned long long begin, end;
      if (std::sscanf(line.c_str(), "%llx-%llx", &begin, &end) == 2)
        active = begin <= reinterpret_cast<uintptr_t>(mapping.data()) && reinterpret_cast<uintptr_t>(mapping.data()) < end;
      if (active && line.rfind("VmFlags:", 0) == 0) {
        std::istringstream flags(line);
        std::string flag;
        bool io = false, pfn = false, writable = false;
        while (flags >> flag) { io |= flag == "io"; pfn |= flag == "pf"; writable |= flag == "wr"; }
        if (!io || !pfn || writable) throw std::runtime_error("resource mapping is not read-only VM_IO/PFNMAP");
        checked = true;
        break;
      }
    }
    if (!checked) throw std::runtime_error("resource mapping absent from smaps");
    std::cout << "DDR read-only VM_IO/PFNMAP mapping/prefault PASS PA=0x" << std::hex
              << resources::index.base + entry.offset << std::dec << " bytes=" << entry.size << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

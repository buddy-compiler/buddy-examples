#pragma once
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using Command = std::array<uint64_t, 4>;

inline bool read_command(Command &command) {
  std::cin.read(reinterpret_cast<char *>(command.data()), sizeof(command));
  if (std::cin.eof() && std::cin.gcount() == 0) return false;
  if (!std::cin) throw std::runtime_error("incomplete command");
  return true;
}

template <typename T> void read_values(T *data, size_t count) {
  std::cin.read(reinterpret_cast<char *>(data), count * sizeof(T));
  if (!std::cin) throw std::runtime_error("incomplete tensor payload");
}

template <typename T> void write_values(const T *data, size_t count) {
  std::cout.write(reinterpret_cast<const char *>(data), count * sizeof(T));
}

#include "execution/state.h"
#include <memory>
#include <interconnect.h>

int main(int argc, char **argv) {
  if (argc != 2) throw std::runtime_error("expected model directory");
  std::cout.exceptions(std::ios::failbit | std::ios::badbit);
  Command command;
  if (!read_command(command)) return 0;
  if (command[0] != 0) throw std::runtime_error("first command must initialize the tile");
  std::unique_ptr<Thinker> thinker;
  if (command[3] == 1) {
    auto directory = std::filesystem::path(argv[1]) / ("chip-" + std::to_string(command[1])) /
                     ("tile-" + std::to_string(command[2]));
    thinker = std::make_unique<Thinker>(directory);
  } else if (command[3] != 0) {
    throw std::runtime_error("unknown tile role");
  }
  const uint64_t ready = 0;
  write_values(&ready, 1);
  std::cout.flush();
  while (read_command(command)) {
    if (command[0] == 11) {
      const uint64_t info[] = {link_read(LINK_CHIP_ID), link_read(LINK_BUFFER_ADDRESS), link_read(LINK_BUFFER_BYTES)};
      write_values(info, 3);
      std::cout.flush();
      continue;
    }
    if (command[0] == 8) {
      std::string text(command[1], '\0');
      read_values(text.data(), text.size());
      std::cerr << text << "\r\n";
      write_values(&ready, 1);
      std::cout.flush();
      continue;
    }
    if (!thinker) throw std::runtime_error("idle tile received a compute request");
    thinker->execute(command);
    std::cout.flush();
  }
}

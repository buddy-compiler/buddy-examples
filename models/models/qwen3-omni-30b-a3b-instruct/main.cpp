#include "execution/audio.h"
#include "execution/state.h"
#include "execution/talker.h"
#include "execution/vision.h"
#include "execution/wave.h"
#include <interconnect.h>
#include <memory>

int main(int argc, char **argv) {
  if (argc != 2)
    throw std::runtime_error("expected model directory");
  std::cout.exceptions(std::ios::failbit | std::ios::badbit);
  Command command;
  if (!read_command(command))
    return 0;
  if (command[0] != 0)
    throw std::runtime_error("first command must initialize the tile");
  std::unique_ptr<Thinker> thinker;
  std::unique_ptr<Vision> vision;
  std::unique_ptr<Audio> audio;
  std::unique_ptr<Talker> talker;
  std::unique_ptr<Wave> wave;
  if (command[3] == 1) {
    auto directory = std::filesystem::path(argv[1]) /
                     ("chip-" + std::to_string(command[1])) /
                     ("tile-" + std::to_string(command[2]));
    thinker = std::make_unique<Thinker>(directory);
  } else if (command[3] == 2) {
    auto directory = std::filesystem::path(argv[1]) /
                     ("chip-" + std::to_string(command[1])) /
                     ("tile-" + std::to_string(command[2]));
    vision = std::make_unique<Vision>(directory);
  } else if (command[3] == 3) {
    auto directory = std::filesystem::path(argv[1]) /
                     ("chip-" + std::to_string(command[1])) /
                     ("tile-" + std::to_string(command[2]));
    audio = std::make_unique<Audio>(directory);
  } else if (command[3] == 4) {
    auto directory = std::filesystem::path(argv[1]) /
                     ("chip-" + std::to_string(command[1])) /
                     ("tile-" + std::to_string(command[2]));
    talker = std::make_unique<Talker>(directory);
  } else if (command[3] == 5) {
    auto directory = std::filesystem::path(argv[1]) /
                     ("chip-" + std::to_string(command[1])) /
                     ("tile-" + std::to_string(command[2]));
    wave = std::make_unique<Wave>(directory);
  } else if (command[3] != 0) {
    throw std::runtime_error("unknown tile role");
  }
  const uint64_t ready = 0;
  write_values(&ready, 1);
  std::cout.flush();
  while (read_command(command)) {
    if (command[0] == 11) {
      const uint64_t info[] = {link_read(LINK_CHIP_ID),
                               link_read(LINK_BUFFER_ADDRESS),
                               link_read(LINK_BUFFER_BYTES)};
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
    if (vision) {
      vision->execute(command);
      std::cout.flush();
      continue;
    }
    if (audio) {
      audio->execute(command);
      std::cout.flush();
      continue;
    }
    if (talker) {
      talker->execute(command);
      std::cout.flush();
      continue;
    }
    if (wave) {
      wave->execute(command);
      std::cout.flush();
      continue;
    }
    if (!thinker)
      throw std::runtime_error("idle tile received a compute request");
    thinker->execute(command);
    std::cout.flush();
  }
}

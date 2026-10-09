#include "model.h"
#include <cstring>
#include <sched.h>

int main(int argc, char **argv) {
  if (argc != 3 && argc != 4)
    throw std::runtime_error(
        "usage: laya-run MODEL_DIRECTORY INPUT_RESOURCE [RESOURCE_INDEX]");
  const std::filesystem::path directory = std::filesystem::absolute(argv[1]);
  resources::configure(directory, argc == 4 ? argv[3] : nullptr);
  const std::filesystem::path inputPath = directory / argv[2];
  size_t bytes = resourceBytes(inputPath);
  resources::Mapping input(inputPath, bytes, alignof(uint64_t));
  if (bytes < 40)
    throw std::runtime_error("incomplete Laya input header");
  const uint64_t *header = static_cast<const uint64_t *>(input.data());
  const auto &shape = modelShape();
  double temperature;
  std::memcpy(&temperature, header + 4, sizeof(temperature));
  if (header[0] != shape.length || header[1] != shape.options ||
      header[2] != shape.actions || header[3] == 0 ||
      bytes !=
          40 + (2 * shape.length + 2 * shape.options + 1) * sizeof(int64_t))
    throw std::runtime_error("Laya input differs from compiled shape");
  cpu_set_t cpus;
  CPU_ZERO(&cpus);
  CPU_SET(controlCpu(0), &cpus);
  if (sched_setaffinity(0, sizeof(cpus), &cpus))
    throw std::system_error(errno, std::generic_category(),
                            "pin Laya main CPU");
  Context context(shape.length, shape.width, shape.options, shape.actions);
  const int64_t *data = reinterpret_cast<const int64_t *>(header + 5);
  for (MemRef<int64_t, 2> *tensor : {static_cast<Tokens *>(&context.tokens),
                                     static_cast<Tokens *>(&context.mask),
                                     static_cast<Tokens *>(&context.positions),
                                     static_cast<Tokens *>(&context.valid)}) {
    std::memcpy(tensor->getData(), data, tensor->getSize() * sizeof(int64_t));
    data += tensor->getSize();
  }
  context.qtype.getData()[0] = *data;
  if (*data < 0 || *data > 2)
    throw std::runtime_error("invalid Laya question type");
  execute(context, directory, header[3]);
  decision(context, shape.options, shape.actions, temperature);
}

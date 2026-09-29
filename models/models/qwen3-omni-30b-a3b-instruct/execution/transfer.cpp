#include "state.h"
#include <interconnect.h>
#include <cstring>

void Thinker::send(size_t count, size_t chip, uint64_t destination) {
  size_t size = count * hiddenSize * sizeof(float);
  if (rank != 0 || exchange.size() != count * hiddenSize || size > link_read(LINK_BUFFER_BYTES))
    throw std::runtime_error("invalid pipeline transfer buffer");
  std::memcpy(reinterpret_cast<void *>(CHIP_LINK_BUFFER), exchange.data(), size);
  if (chip_send(chip, link_read(LINK_BUFFER_ADDRESS), destination, size, 1))
    throw std::runtime_error("pipeline DMA failed");
  const uint64_t ready = 0;
  write_values(&ready, 1);
}

void Thinker::receive(size_t count, size_t source) {
  size_t size = count * hiddenSize * sizeof(float);
  if (rank != 0 || size > link_read(LINK_BUFFER_BYTES))
    throw std::runtime_error("invalid pipeline receive buffer");
  while (!link_read(LINK_EVENT_COUNT)) {}
  if (link_read(LINK_EVENT_SOURCE) != source || link_read(LINK_EVENT_TAG) != 1 ||
      link_read(LINK_EVENT_BYTES) != size)
    throw std::runtime_error("unexpected pipeline DMA completion");
  __asm__ volatile("fence rw,rw" ::: "memory");
  write_values(reinterpret_cast<const float *>(CHIP_LINK_BUFFER), count * hiddenSize);
  link_write(LINK_EVENT_ACK, 1);
}

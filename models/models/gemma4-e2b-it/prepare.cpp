#include <buddy/LLM/TextContainer.h>
#include <cstdint>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
  if (argc != 3)
    throw std::runtime_error("usage: prepare VOCAB PROMPT");
  buddy::Text<size_t, 2> input(argv[2]);
  input.tokenizeGemma4(argv[1], 512);
  const uint64_t count = input.getTokenCnt();
  for (uint64_t index = 0; index <= count; ++index) {
    uint64_t value = index ? input.getData()[index - 1] : count;
    char bytes[8];
    for (unsigned byte = 0; byte < 8; ++byte)
      bytes[byte] = char(value >> (byte * 8));
    std::cout.write(bytes, sizeof(bytes));
  }
}

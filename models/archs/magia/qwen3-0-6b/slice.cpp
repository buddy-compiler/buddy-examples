// A slice of Qwen3-0.6B small enough for an RTL simulation of the MAGIA
// device: the first matrix product of the first decode step, the query
// projection of layer 0 for one token and its first 64 output features
// (every term: these are the model's own values). Its FP16
// operands and result are those of the reference run of the model
// (buddy-pytorch-importer.py, the products as the mesh computes them); Loom
// hands it to the mesh as the compiled model does and compares the result
// bit for bit.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern "C" void magia_gemm(uint64_t x, uint64_t w, uint64_t z, uint64_t m,
                           uint64_t n, uint64_t k);

constexpr unsigned rows = 1, terms = 1024, columns = 64;

alignas(64) static const unsigned char x_data[] = {
#embed "slice-x.data"
};
alignas(64) static const unsigned char w_data[] = {
#embed "slice-w.data"
};
alignas(64) static const unsigned char z_data[] = {
#embed "slice-z.data"
};
static_assert(sizeof(x_data) == rows * terms * 2);
static_assert(sizeof(w_data) == terms * columns * 2);
static_assert(sizeof(z_data) == rows * columns * 2);
alignas(64) static uint16_t z[rows * columns];
static_assert(rows * columns % 4 == 0);

static uint64_t cycles() {
  uint64_t value;
  asm volatile("rdcycle %0" : "=r"(value));
  return value;
}

int main() {
  const uint64_t start = cycles();
  magia_gemm((uintptr_t)x_data, (uintptr_t)w_data, (uintptr_t)z, rows, terms,
             columns);
  const uint64_t elapsed = cycles() - start;
  const auto *expected = reinterpret_cast<const uint16_t *>(z_data);
  // Loom takes about 13 cycles an instruction: compare four values a word,
  // and look at the values of a word only where it differs.
  const auto *words = reinterpret_cast<const uint64_t *>(z);
  const auto *expected_words = reinterpret_cast<const uint64_t *>(z_data);
  unsigned wrong = 0;
  for (unsigned word = 0; word < rows * columns / 4; ++word) {
    if (words[word] == expected_words[word])
      continue;
    for (unsigned i = 4 * word; i < 4 * word + 4; ++i)
      if (z[i] != expected[i]) {
        if (!wrong)
          printf("qwen3 slice: row %u column %u is %04x, not %04x\n",
                 i / columns, i % columns, z[i], expected[i]);
        ++wrong;
      }
  }
  printf("qwen3 slice (layer 0 query, a decode step, 64 features) on the MAGIA "
         "device: %u "
         "of %u "
         "values as the model's, %lu cycles\n",
         rows * columns - wrong, rows * columns, (unsigned long)elapsed);
  return wrong != 0;
}

// pi0-FAST on the piplus device, baremetal: every hart enters main; the
// control hart reads the rows of the embedding table of the tokens for the
// programs (the table is in the model image once, in no graph). It
// runs the prefill program (the cameras and the prompt to the
// first token and the keys and values of every layer), then the decoding
// step program once per token until the EOS, keeping between the steps what
// the next reads: the keys and values of the tokens generated so far and the
// bias of the scores, which sees them. It checks the tokens against those of
// the quantized program in PyTorch (generated.data).
#include "ModelImage.h"
#include "model.h"
#include "testutils.h"
#include <cstring>
#include <device.h>

extern "C" int prefill_program_job(void *, void *);
extern "C" int decode_program_job(void *, void *);

#define KEYS (PREFIX_KEYS + GENERATED + OWN_KEYS)
static float logits[VOCABULARY] __attribute__((aligned(64)));
static float bias[KEYS] __attribute__((aligned(64)));
// The keys of every layer, then their values: the prefill's, those of the
// generated tokens (a row a token) and a step's own.
static float prefix[2 * LAYERS][PREFIX_KEYS * HEAD_DIM]
    __attribute__((aligned(64)));
static float before[2 * LAYERS][GENERATED * HEAD_DIM]
    __attribute__((aligned(64)));
static float own[2 * LAYERS][HEAD_DIM] __attribute__((aligned(64)));
static float prompt[201 * HIDDEN] __attribute__((aligned(64)));
static float row[HIDDEN] __attribute__((aligned(64)));
static int64_t position, token, place;
static int64_t generated[GENERATED];

// The final output on the SCU's output stream of the control hart, for the
// checks of the simulations: the first run captures it, later runs compare it.
static const uint8_t reference[sizeof(generated)]
    __attribute__((section(".bbsim.reference"), used)) = {};
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} capture __attribute__((section(".bbsim.capture_output"), used, retain)) = {
    nullptr, reference, sizeof(reference)};

static void stream_output(uintptr_t hart) {
  auto stream =
      reinterpret_cast<volatile uint32_t *>(0x60000000ULL + hart * 0x40000);
  *stream = 0x42424f55;
  for (unsigned i = 0; i < sizeof(generated) / 4; ++i) {
    uint32_t word;
    memcpy(&word, reinterpret_cast<const uint8_t *>(generated) + 4 * i,
           sizeof(word));
    *stream = word;
  }
}

// The most likely token.
static int64_t likeliest() {
  int64_t best = 0;
  for (int64_t i = 1; i < VOCABULARY; ++i)
    if (logits[i] > logits[best])
      best = i;
  return best;
}

int main() {
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  // The other harts run their entries of every run of the control hart.
  if (hart != DEVICE_CONTROL_HART)
    for (;;)
      program_run(nullptr, nullptr);
  uint64_t bytes;
  const char *files[] = {"ddr/vision-arg0.data",     "ddr/vision-arg1.data",
                         "ddr/prefix-pack-f32.data", "ddr/prefix-pack-i8.data",
                         "ddr/step-pack-f32.data",   "ddr/step-pack-i8.data"};
  void *packs[6];
  for (unsigned i = 0; i < 6; ++i)
    packs[i] =
        const_cast<char *>(model::imageFile(MODEL_IMAGE, files[i], &bytes));
  const auto &input = *reinterpret_cast<const Observation *>(
      model::imageFile(MODEL_IMAGE, "observation.data", &bytes));
  const auto *table = reinterpret_cast<const float *>(
      model::imageFile(MODEL_IMAGE, "ddr/embedding.data", &bytes));
  for (unsigned i = 0; i < 201; ++i)
    memcpy(prompt + i * HIDDEN, table + input.tokens[i] * HIDDEN, sizeof(row));

  // The prefill: its packs, the images, the rows of the prompt and its mask;
  // the logits of the first token, the position of the next, the bias and the
  // keys and values.
  static void *prefill[7 + 3 + 2 * LAYERS];
  for (unsigned i = 0; i < 4; ++i)
    prefill[i] = packs[i];
  prefill[4] = const_cast<float *>(input.images);
  prefill[5] = prompt;
  prefill[6] = const_cast<int64_t *>(input.masks);
  prefill[7] = logits;
  prefill[8] = &position;
  prefill[9] = bias;
  for (unsigned i = 0; i < 2 * LAYERS; ++i)
    prefill[10 + i] = prefix[i];
  const uint64_t begin = read_counter();
  if (program_run(prefill_program_job, prefill)) {
    printf("prefill program failed\n");
    return 1;
  }
  const uint64_t prefilled = read_counter() - begin;
  unsigned count = 0;
  token = likeliest();
  generated[count++] = token;

  // A decoding step: its packs, the row of the token before, its position, the
  // bias, the keys and values; the logits of the next token and its own keys
  // and values.
  static void *decode[2 + 3 + 4 * LAYERS + 1 + 2 * LAYERS];
  decode[0] = packs[4];
  decode[1] = packs[5];
  decode[2] = row;
  decode[3] = &place;
  decode[4] = bias;
  for (unsigned i = 0; i < 2 * LAYERS; ++i) {
    decode[5 + i] = prefix[i];
    decode[5 + 2 * LAYERS + i] = before[i];
    decode[6 + 4 * LAYERS + i] = own[i];
  }
  decode[5 + 4 * LAYERS] = logits;
  for (unsigned step = 0; step < GENERATED - 1 && token != EOS; ++step) {
    place = position + step;
    memcpy(row, table + token * HIDDEN, sizeof(row));
    const int status = program_run(decode_program_job, decode);
    if (status) {
      printf("decode program failed at token %u: %d\n", count, status);
      return 1;
    }
    // The token's keys and values after the generated ones, seen from the
    // next step on.
    for (unsigned i = 0; i < 2 * LAYERS; ++i)
      memcpy(before[i] + step * HEAD_DIM, own[i], sizeof(own[i]));
    bias[PREFIX_KEYS + step] = 0.0f;
    token = likeliest();
    generated[count++] = token;
  }
  const uint64_t elapsed = read_counter() - begin;
  printf("Tokens: %u\nPrefill cycles: %lu\nInference cycles: %lu\n", count,
         static_cast<unsigned long>(prefilled),
         static_cast<unsigned long>(elapsed));
  // The tokens against PyTorch (generated.data): its tokens up to the EOS.
  const auto *expected = reinterpret_cast<const int64_t *>(
      model::imageFile(MODEL_IMAGE, "generated.data", &bytes));
  unsigned different = 0, length = 0;
  while (length < GENERATED && expected[length] != 0)
    ++length;
  for (unsigned i = 0; i < count && i < length; ++i)
    different += generated[i] != expected[i];
  printf("tokens: %u generated, %u in PyTorch, %u of the first %u differ\n",
         count, length, different, count < length ? count : length);
  stream_output(hart);
  return 0;
}

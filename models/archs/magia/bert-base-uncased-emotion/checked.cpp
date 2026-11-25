#include "checked.h"
// BERT (emotion) on the MAGIA device: Loom runs the compiled graph bare
// metal, the mesh its matrix products. The first four validation sentences
// against the model in PyTorch with the mesh's products in software
// (buddy-pytorch-importer.py); the recipe's check on this machine runs all
// of them, Loom takes about ten billion cycles a sentence.
#include "testutils.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// A memref of the compiled graph.
template <typename T, int N> struct Buffer {
  const T *allocated;
  const T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};

// The FP16 weights and the FP32 parameters; the imported ABI orders inputs as
// IDs, positions, types, then mask.
extern "C" void
_mlir_ciface_forward(Buffer<float, 2> *output, Buffer<uint16_t, 1> *weights,
                     Buffer<float, 1> *parameters, Buffer<int64_t, 2> *ids,
                     Buffer<int64_t, 2> *positions, Buffer<int64_t, 2> *types,
                     Buffer<int64_t, 2> *mask);

alignas(8) static const unsigned char type_data[] = {
#embed "check-types.data"
};
alignas(8) static const unsigned char mask_data[] = {
#embed "check-mask.data"
};
alignas(8) static const unsigned char position_data[] = {
#embed "check-positions.data"
};
alignas(8) static const unsigned char logit_data[] = {
#embed "check-logits.data"
};

// The logits of every check sentence, compared at once at the end.
static float outputs[4 * 6];

static Buffer<int64_t, 2> sentence(const unsigned char *data, unsigned sample) {
  const auto *values = reinterpret_cast<const int64_t *>(data) + sample * 32;
  return {values, values, 0, {1, 32}, {32, 1}};
}

int check_model(const uint16_t *halves, intptr_t weight_count,
                const float *floats, intptr_t parameter_count,
                const int64_t *input, unsigned input_count) {
  const auto *logits = reinterpret_cast<const float *>(logit_data);
  Buffer<uint16_t, 1> weights{halves, halves, 0, {weight_count}, {1}};
  Buffer<float, 1> parameters{floats, floats, 0, {parameter_count}, {1}};
  const unsigned sentences = 4;
  if (input_count < sentences * 32)
    return 1;
  unsigned same = 0;
  uint64_t total = 0;
  for (unsigned sample = 0; sample < sentences; ++sample) {
    auto ids = sentence(reinterpret_cast<const unsigned char *>(input), sample);
    auto types = sentence(type_data, sample);
    auto mask = sentence(mask_data, sample);
    auto positions = sentence(position_data, sample);
    // Each invocation returns an owned output descriptor.
    Buffer<float, 2> output;
    const uint64_t start = read_counter();
    _mlir_ciface_forward(&output, &weights, &parameters, &ids, &positions,
                         &types, &mask);
    const uint64_t elapsed = read_counter() - start;
    total += elapsed;
    const float *got = output.aligned + output.offset;
    const float *expected = logits + sample * 6;
    unsigned best = 0, reference = 0;
    for (unsigned i = 0; i < 6; ++i) {
      if (got[i] > got[best])
        best = i;
      if (expected[i] > expected[reference])
        reference = i;
      outputs[sample * 6 + i] = got[i];
    }
    same += best == reference;
    printf("Sentence %u: class %u (the reference %u), %lu cycles\n", sample,
           best, reference, static_cast<unsigned long>(elapsed));
    free(const_cast<float *>(output.allocated));
  }
  printf("Inference cycles: %lu\n",
         static_cast<unsigned long>(total / sentences));
  // The reference computes the mesh's FP16 products the same way, but the
  // other operations round differently in FP32, which flips FP16 roundings
  // of the inputs of later products (2^-11 each) over the 12 layers: the
  // compiled model on the host (bert-base-uncased-emotion-model-check)
  // stays within 0.75% of the largest logit on the 64 check sentences.
  const int failed = check_output(outputs, logits, sentences * 6, 1e-2f);
  printf("Top-1: %u/%u as the reference\n", same, sentences);
  return failed || same != sentences;
}

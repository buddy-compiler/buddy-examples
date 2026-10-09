#pragma once
#include "gemma-parameters.h"
#include <array>
#include <buddy/Core/Container.h>
#include <cstdint>
#include <vector>

template <typename T, size_t N> struct Tensor : MemRef<T, N> {
  using MemRef<T, N>::MemRef;
  void *storage() const { return this->allocated; }
};

struct CacheLayer {
  Tensor<long long, 1> length;
  Tensor<int8_t, 4> keyCodes, keyScales, valueCodes, valueScales;
  CacheLayer(size_t width, bool allocate)
      : length({1}, allocate, 0),
        keyCodes({1, 1, CacheLength, width}, allocate, 0),
        keyScales({1, 1, CacheLength, width / 32}, allocate, 0),
        valueCodes({1, 1, CacheLength, width}, allocate, 0),
        valueScales({1, 1, CacheLength, width / 32}, allocate, 0) {
    if (allocate) {
      length.getData()[0] = 0;
      std::fill_n(keyCodes.getData(), keyCodes.getSize(), int8_t(0));
      std::fill_n(valueCodes.getData(), valueCodes.getSize(), int8_t(0));
      std::fill_n(keyScales.getData(), keyScales.getSize(), int8_t(127));
      std::fill_n(valueScales.getData(), valueScales.getSize(), int8_t(127));
    }
  }
};

struct MemRefContainer {
  std::array<CacheLayer, 15> layers;
  Tensor<float, 3> logits;
  MemRefContainer(size_t tokens, bool allocate)
      : layers{CacheLayer(256, allocate), CacheLayer(256, allocate),
               CacheLayer(256, allocate), CacheLayer(256, allocate),
               CacheLayer(512, allocate), CacheLayer(256, allocate),
               CacheLayer(256, allocate), CacheLayer(256, allocate),
               CacheLayer(256, allocate), CacheLayer(512, allocate),
               CacheLayer(256, allocate), CacheLayer(256, allocate),
               CacheLayer(256, allocate), CacheLayer(256, allocate),
               CacheLayer(512, allocate)},
        logits({1, tokens, 262144}, false, 0) {}
  std::vector<void *> allocations() {
    std::vector<void *> result;
    for (auto &layer : layers) {
      result.push_back(layer.length.storage());
      result.push_back(layer.keyCodes.storage());
      result.push_back(layer.keyScales.storage());
      result.push_back(layer.valueCodes.storage());
      result.push_back(layer.valueScales.storage());
    }
    result.push_back(logits.storage());
    return result;
  }
  void detach() {
    for (auto &layer : layers) {
      layer.length.release();
      layer.keyCodes.release();
      layer.keyScales.release();
      layer.valueCodes.release();
      layer.valueScales.release();
    }
    logits.release();
  }
};

static_assert(sizeof(CacheLayer) == 49 * sizeof(intptr_t));

extern "C" void _mlir_ciface_forward_prefill(
    MemRefContainer *result, MemRef<float, 1> *params,
    MemRef<int8_t, 1> *weights, MemRef<long long, 2> *input_ids,
    MemRef<long long, 2> *position_ids, MemRef<long long, 1> *length0,
    MemRef<int8_t, 4> *keyCodes0, MemRef<int8_t, 4> *keyScales0,
    MemRef<int8_t, 4> *valueCodes0, MemRef<int8_t, 4> *valueScales0,
    MemRef<long long, 1> *length1, MemRef<int8_t, 4> *keyCodes1,
    MemRef<int8_t, 4> *keyScales1, MemRef<int8_t, 4> *valueCodes1,
    MemRef<int8_t, 4> *valueScales1, MemRef<long long, 1> *length2,
    MemRef<int8_t, 4> *keyCodes2, MemRef<int8_t, 4> *keyScales2,
    MemRef<int8_t, 4> *valueCodes2, MemRef<int8_t, 4> *valueScales2,
    MemRef<long long, 1> *length3, MemRef<int8_t, 4> *keyCodes3,
    MemRef<int8_t, 4> *keyScales3, MemRef<int8_t, 4> *valueCodes3,
    MemRef<int8_t, 4> *valueScales3, MemRef<long long, 1> *length4,
    MemRef<int8_t, 4> *keyCodes4, MemRef<int8_t, 4> *keyScales4,
    MemRef<int8_t, 4> *valueCodes4, MemRef<int8_t, 4> *valueScales4,
    MemRef<long long, 1> *length5, MemRef<int8_t, 4> *keyCodes5,
    MemRef<int8_t, 4> *keyScales5, MemRef<int8_t, 4> *valueCodes5,
    MemRef<int8_t, 4> *valueScales5, MemRef<long long, 1> *length6,
    MemRef<int8_t, 4> *keyCodes6, MemRef<int8_t, 4> *keyScales6,
    MemRef<int8_t, 4> *valueCodes6, MemRef<int8_t, 4> *valueScales6,
    MemRef<long long, 1> *length7, MemRef<int8_t, 4> *keyCodes7,
    MemRef<int8_t, 4> *keyScales7, MemRef<int8_t, 4> *valueCodes7,
    MemRef<int8_t, 4> *valueScales7, MemRef<long long, 1> *length8,
    MemRef<int8_t, 4> *keyCodes8, MemRef<int8_t, 4> *keyScales8,
    MemRef<int8_t, 4> *valueCodes8, MemRef<int8_t, 4> *valueScales8,
    MemRef<long long, 1> *length9, MemRef<int8_t, 4> *keyCodes9,
    MemRef<int8_t, 4> *keyScales9, MemRef<int8_t, 4> *valueCodes9,
    MemRef<int8_t, 4> *valueScales9, MemRef<long long, 1> *length10,
    MemRef<int8_t, 4> *keyCodes10, MemRef<int8_t, 4> *keyScales10,
    MemRef<int8_t, 4> *valueCodes10, MemRef<int8_t, 4> *valueScales10,
    MemRef<long long, 1> *length11, MemRef<int8_t, 4> *keyCodes11,
    MemRef<int8_t, 4> *keyScales11, MemRef<int8_t, 4> *valueCodes11,
    MemRef<int8_t, 4> *valueScales11, MemRef<long long, 1> *length12,
    MemRef<int8_t, 4> *keyCodes12, MemRef<int8_t, 4> *keyScales12,
    MemRef<int8_t, 4> *valueCodes12, MemRef<int8_t, 4> *valueScales12,
    MemRef<long long, 1> *length13, MemRef<int8_t, 4> *keyCodes13,
    MemRef<int8_t, 4> *keyScales13, MemRef<int8_t, 4> *valueCodes13,
    MemRef<int8_t, 4> *valueScales13, MemRef<long long, 1> *length14,
    MemRef<int8_t, 4> *keyCodes14, MemRef<int8_t, 4> *keyScales14,
    MemRef<int8_t, 4> *valueCodes14, MemRef<int8_t, 4> *valueScales14);

extern "C" void _mlir_ciface_forward_decode(
    MemRefContainer *result, MemRef<float, 1> *params,
    MemRef<int8_t, 1> *weights, MemRef<long long, 2> *input_ids,
    MemRef<long long, 2> *position_ids, MemRef<long long, 1> *length0,
    MemRef<int8_t, 4> *keyCodes0, MemRef<int8_t, 4> *keyScales0,
    MemRef<int8_t, 4> *valueCodes0, MemRef<int8_t, 4> *valueScales0,
    MemRef<long long, 1> *length1, MemRef<int8_t, 4> *keyCodes1,
    MemRef<int8_t, 4> *keyScales1, MemRef<int8_t, 4> *valueCodes1,
    MemRef<int8_t, 4> *valueScales1, MemRef<long long, 1> *length2,
    MemRef<int8_t, 4> *keyCodes2, MemRef<int8_t, 4> *keyScales2,
    MemRef<int8_t, 4> *valueCodes2, MemRef<int8_t, 4> *valueScales2,
    MemRef<long long, 1> *length3, MemRef<int8_t, 4> *keyCodes3,
    MemRef<int8_t, 4> *keyScales3, MemRef<int8_t, 4> *valueCodes3,
    MemRef<int8_t, 4> *valueScales3, MemRef<long long, 1> *length4,
    MemRef<int8_t, 4> *keyCodes4, MemRef<int8_t, 4> *keyScales4,
    MemRef<int8_t, 4> *valueCodes4, MemRef<int8_t, 4> *valueScales4,
    MemRef<long long, 1> *length5, MemRef<int8_t, 4> *keyCodes5,
    MemRef<int8_t, 4> *keyScales5, MemRef<int8_t, 4> *valueCodes5,
    MemRef<int8_t, 4> *valueScales5, MemRef<long long, 1> *length6,
    MemRef<int8_t, 4> *keyCodes6, MemRef<int8_t, 4> *keyScales6,
    MemRef<int8_t, 4> *valueCodes6, MemRef<int8_t, 4> *valueScales6,
    MemRef<long long, 1> *length7, MemRef<int8_t, 4> *keyCodes7,
    MemRef<int8_t, 4> *keyScales7, MemRef<int8_t, 4> *valueCodes7,
    MemRef<int8_t, 4> *valueScales7, MemRef<long long, 1> *length8,
    MemRef<int8_t, 4> *keyCodes8, MemRef<int8_t, 4> *keyScales8,
    MemRef<int8_t, 4> *valueCodes8, MemRef<int8_t, 4> *valueScales8,
    MemRef<long long, 1> *length9, MemRef<int8_t, 4> *keyCodes9,
    MemRef<int8_t, 4> *keyScales9, MemRef<int8_t, 4> *valueCodes9,
    MemRef<int8_t, 4> *valueScales9, MemRef<long long, 1> *length10,
    MemRef<int8_t, 4> *keyCodes10, MemRef<int8_t, 4> *keyScales10,
    MemRef<int8_t, 4> *valueCodes10, MemRef<int8_t, 4> *valueScales10,
    MemRef<long long, 1> *length11, MemRef<int8_t, 4> *keyCodes11,
    MemRef<int8_t, 4> *keyScales11, MemRef<int8_t, 4> *valueCodes11,
    MemRef<int8_t, 4> *valueScales11, MemRef<long long, 1> *length12,
    MemRef<int8_t, 4> *keyCodes12, MemRef<int8_t, 4> *keyScales12,
    MemRef<int8_t, 4> *valueCodes12, MemRef<int8_t, 4> *valueScales12,
    MemRef<long long, 1> *length13, MemRef<int8_t, 4> *keyCodes13,
    MemRef<int8_t, 4> *keyScales13, MemRef<int8_t, 4> *valueCodes13,
    MemRef<int8_t, 4> *valueScales13, MemRef<long long, 1> *length14,
    MemRef<int8_t, 4> *keyCodes14, MemRef<int8_t, 4> *keyScales14,
    MemRef<int8_t, 4> *valueCodes14, MemRef<int8_t, 4> *valueScales14);

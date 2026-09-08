//===- buddy-qwen3-0.6b-main.cpp
//-------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include <array>
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/time.h>

using namespace buddy;
double total_time = 0;
constexpr size_t ParamsBf16Size = 596049920;
constexpr size_t ParamsF32Size = 64;
constexpr size_t MaxVocabSize = 151936;
constexpr size_t MaxTokenLength = 16;

constexpr size_t NUM_LAYERS = 56;
constexpr size_t HiddenSize = 128;
constexpr size_t HeadNum = 8;

extern "C" double _mlir_ciface_rtclock() {
#ifndef _WIN32
  struct timeval tp;
  int stat = gettimeofday(&tp, nullptr);
  if (stat != 0)
    fprintf(stderr, "Error returning time from gettimeofday: %d\n", stat);
  return (tp.tv_sec + tp.tv_usec * 1.0e-6);
#else
  fprintf(stderr, "Timing utility not implemented on Windows\n");
  return 0.0;
#endif // _WIN32
}

struct MemRefContainer {
  // Layout must match MLIR returns: (i64, k, v) x 28 + logits.
  MemRef<long long, 1> pos0;
  MemRef<uint16_t, 4> kv0;
  MemRef<uint16_t, 4> kv1;
  MemRef<long long, 1> pos1;
  MemRef<uint16_t, 4> kv2;
  MemRef<uint16_t, 4> kv3;
  MemRef<long long, 1> pos2;
  MemRef<uint16_t, 4> kv4;
  MemRef<uint16_t, 4> kv5;
  MemRef<long long, 1> pos3;
  MemRef<uint16_t, 4> kv6;
  MemRef<uint16_t, 4> kv7;
  MemRef<long long, 1> pos4;
  MemRef<uint16_t, 4> kv8;
  MemRef<uint16_t, 4> kv9;
  MemRef<long long, 1> pos5;
  MemRef<uint16_t, 4> kv10;
  MemRef<uint16_t, 4> kv11;
  MemRef<long long, 1> pos6;
  MemRef<uint16_t, 4> kv12;
  MemRef<uint16_t, 4> kv13;
  MemRef<long long, 1> pos7;
  MemRef<uint16_t, 4> kv14;
  MemRef<uint16_t, 4> kv15;
  MemRef<long long, 1> pos8;
  MemRef<uint16_t, 4> kv16;
  MemRef<uint16_t, 4> kv17;
  MemRef<long long, 1> pos9;
  MemRef<uint16_t, 4> kv18;
  MemRef<uint16_t, 4> kv19;
  MemRef<long long, 1> pos10;
  MemRef<uint16_t, 4> kv20;
  MemRef<uint16_t, 4> kv21;
  MemRef<long long, 1> pos11;
  MemRef<uint16_t, 4> kv22;
  MemRef<uint16_t, 4> kv23;
  MemRef<long long, 1> pos12;
  MemRef<uint16_t, 4> kv24;
  MemRef<uint16_t, 4> kv25;
  MemRef<long long, 1> pos13;
  MemRef<uint16_t, 4> kv26;
  MemRef<uint16_t, 4> kv27;
  MemRef<long long, 1> pos14;
  MemRef<uint16_t, 4> kv28;
  MemRef<uint16_t, 4> kv29;
  MemRef<long long, 1> pos15;
  MemRef<uint16_t, 4> kv30;
  MemRef<uint16_t, 4> kv31;
  MemRef<long long, 1> pos16;
  MemRef<uint16_t, 4> kv32;
  MemRef<uint16_t, 4> kv33;
  MemRef<long long, 1> pos17;
  MemRef<uint16_t, 4> kv34;
  MemRef<uint16_t, 4> kv35;
  MemRef<long long, 1> pos18;
  MemRef<uint16_t, 4> kv36;
  MemRef<uint16_t, 4> kv37;
  MemRef<long long, 1> pos19;
  MemRef<uint16_t, 4> kv38;
  MemRef<uint16_t, 4> kv39;
  MemRef<long long, 1> pos20;
  MemRef<uint16_t, 4> kv40;
  MemRef<uint16_t, 4> kv41;
  MemRef<long long, 1> pos21;
  MemRef<uint16_t, 4> kv42;
  MemRef<uint16_t, 4> kv43;
  MemRef<long long, 1> pos22;
  MemRef<uint16_t, 4> kv44;
  MemRef<uint16_t, 4> kv45;
  MemRef<long long, 1> pos23;
  MemRef<uint16_t, 4> kv46;
  MemRef<uint16_t, 4> kv47;
  MemRef<long long, 1> pos24;
  MemRef<uint16_t, 4> kv48;
  MemRef<uint16_t, 4> kv49;
  MemRef<long long, 1> pos25;
  MemRef<uint16_t, 4> kv50;
  MemRef<uint16_t, 4> kv51;
  MemRef<long long, 1> pos26;
  MemRef<uint16_t, 4> kv52;
  MemRef<uint16_t, 4> kv53;
  MemRef<long long, 1> pos27;
  MemRef<uint16_t, 4> kv54;
  MemRef<uint16_t, 4> kv55;
  MemRef<uint16_t, 3> logits;

  MemRefContainer(
      MemRef<uint16_t, 4> k0, MemRef<uint16_t, 4> k1, MemRef<uint16_t, 4> k2, MemRef<uint16_t, 4> k3,
      MemRef<uint16_t, 4> k4, MemRef<uint16_t, 4> k5, MemRef<uint16_t, 4> k6, MemRef<uint16_t, 4> k7,
      MemRef<uint16_t, 4> k8, MemRef<uint16_t, 4> k9, MemRef<uint16_t, 4> k10, MemRef<uint16_t, 4> k11,
      MemRef<uint16_t, 4> k12, MemRef<uint16_t, 4> k13, MemRef<uint16_t, 4> k14, MemRef<uint16_t, 4> k15,
      MemRef<uint16_t, 4> k16, MemRef<uint16_t, 4> k17, MemRef<uint16_t, 4> k18, MemRef<uint16_t, 4> k19,
      MemRef<uint16_t, 4> k20, MemRef<uint16_t, 4> k21, MemRef<uint16_t, 4> k22, MemRef<uint16_t, 4> k23,
      MemRef<uint16_t, 4> k24, MemRef<uint16_t, 4> k25, MemRef<uint16_t, 4> k26, MemRef<uint16_t, 4> k27,
      MemRef<uint16_t, 4> k28, MemRef<uint16_t, 4> k29, MemRef<uint16_t, 4> k30, MemRef<uint16_t, 4> k31,
      MemRef<uint16_t, 4> k32, MemRef<uint16_t, 4> k33, MemRef<uint16_t, 4> k34, MemRef<uint16_t, 4> k35,
      MemRef<uint16_t, 4> k36, MemRef<uint16_t, 4> k37, MemRef<uint16_t, 4> k38, MemRef<uint16_t, 4> k39,
      MemRef<uint16_t, 4> k40, MemRef<uint16_t, 4> k41, MemRef<uint16_t, 4> k42, MemRef<uint16_t, 4> k43,
      MemRef<uint16_t, 4> k44, MemRef<uint16_t, 4> k45, MemRef<uint16_t, 4> k46, MemRef<uint16_t, 4> k47,
      MemRef<uint16_t, 4> k48, MemRef<uint16_t, 4> k49, MemRef<uint16_t, 4> k50, MemRef<uint16_t, 4> k51,
      MemRef<uint16_t, 4> k52, MemRef<uint16_t, 4> k53, MemRef<uint16_t, 4> k54, MemRef<uint16_t, 4> k55,
      MemRef<uint16_t, 3> l)
      : pos0({1}, 0LL),
        kv0(k0),
        kv1(k1),
        pos1({1}, 0LL),
        kv2(k2),
        kv3(k3),
        pos2({1}, 0LL),
        kv4(k4),
        kv5(k5),
        pos3({1}, 0LL),
        kv6(k6),
        kv7(k7),
        pos4({1}, 0LL),
        kv8(k8),
        kv9(k9),
        pos5({1}, 0LL),
        kv10(k10),
        kv11(k11),
        pos6({1}, 0LL),
        kv12(k12),
        kv13(k13),
        pos7({1}, 0LL),
        kv14(k14),
        kv15(k15),
        pos8({1}, 0LL),
        kv16(k16),
        kv17(k17),
        pos9({1}, 0LL),
        kv18(k18),
        kv19(k19),
        pos10({1}, 0LL),
        kv20(k20),
        kv21(k21),
        pos11({1}, 0LL),
        kv22(k22),
        kv23(k23),
        pos12({1}, 0LL),
        kv24(k24),
        kv25(k25),
        pos13({1}, 0LL),
        kv26(k26),
        kv27(k27),
        pos14({1}, 0LL),
        kv28(k28),
        kv29(k29),
        pos15({1}, 0LL),
        kv30(k30),
        kv31(k31),
        pos16({1}, 0LL),
        kv32(k32),
        kv33(k33),
        pos17({1}, 0LL),
        kv34(k34),
        kv35(k35),
        pos18({1}, 0LL),
        kv36(k36),
        kv37(k37),
        pos19({1}, 0LL),
        kv38(k38),
        kv39(k39),
        pos20({1}, 0LL),
        kv40(k40),
        kv41(k41),
        pos21({1}, 0LL),
        kv42(k42),
        kv43(k43),
        pos22({1}, 0LL),
        kv44(k44),
        kv45(k45),
        pos23({1}, 0LL),
        kv46(k46),
        kv47(k47),
        pos24({1}, 0LL),
        kv48(k48),
        kv49(k49),
        pos25({1}, 0LL),
        kv50(k50),
        kv51(k51),
        pos26({1}, 0LL),
        kv52(k52),
        kv53(k53),
        pos27({1}, 0LL),
        kv54(k54),
        kv55(k55),
        logits(l) {}
};

using KVPtrArray = std::array<MemRef<uint16_t, 4> *, 56>;

KVPtrArray buildKVPtrs(MemRefContainer &c) {
  return {&c.kv0,  &c.kv1,  &c.kv2,  &c.kv3,  &c.kv4,  &c.kv5,  &c.kv6,  &c.kv7,
          &c.kv8,  &c.kv9,  &c.kv10, &c.kv11, &c.kv12, &c.kv13, &c.kv14, &c.kv15,
          &c.kv16, &c.kv17, &c.kv18, &c.kv19, &c.kv20, &c.kv21, &c.kv22, &c.kv23,
          &c.kv24, &c.kv25, &c.kv26, &c.kv27, &c.kv28, &c.kv29, &c.kv30, &c.kv31,
          &c.kv32, &c.kv33, &c.kv34, &c.kv35, &c.kv36, &c.kv37, &c.kv38, &c.kv39,
          &c.kv40, &c.kv41, &c.kv42, &c.kv43, &c.kv44, &c.kv45, &c.kv46, &c.kv47,
          &c.kv48, &c.kv49, &c.kv50, &c.kv51, &c.kv52, &c.kv53, &c.kv54, &c.kv55};
}

/// Declare Qwen3 forward function.
extern "C" void _mlir_ciface_forward_prefill(
    MemRefContainer *result, MemRef<uint16_t, 1> *arg0, MemRef<float, 1> *arg1,
    Text<size_t, 2> *arg2,
    MemRef<long long, 1> *pos0,
    MemRef<long long, 1> *pos1,
    MemRef<long long, 1> *pos2,
    MemRef<long long, 1> *pos3,
    MemRef<long long, 1> *pos4,
    MemRef<long long, 1> *pos5,
    MemRef<long long, 1> *pos6,
    MemRef<long long, 1> *pos7,
    MemRef<long long, 1> *pos8,
    MemRef<long long, 1> *pos9,
    MemRef<long long, 1> *pos10,
    MemRef<long long, 1> *pos11,
    MemRef<long long, 1> *pos12,
    MemRef<long long, 1> *pos13,
    MemRef<long long, 1> *pos14,
    MemRef<long long, 1> *pos15,
    MemRef<long long, 1> *pos16,
    MemRef<long long, 1> *pos17,
    MemRef<long long, 1> *pos18,
    MemRef<long long, 1> *pos19,
    MemRef<long long, 1> *pos20,
    MemRef<long long, 1> *pos21,
    MemRef<long long, 1> *pos22,
    MemRef<long long, 1> *pos23,
    MemRef<long long, 1> *pos24,
    MemRef<long long, 1> *pos25,
    MemRef<long long, 1> *pos26,
    MemRef<long long, 1> *pos27);

extern "C" void _mlir_ciface_forward_decode(
    MemRefContainer *result, MemRef<uint16_t, 1> *arg0, MemRef<float, 1> *arg1,
    MemRef<long long, 2> *arg2,
    MemRef<long long, 1> *pos0,
    MemRef<uint16_t, 4> *kv0,
    MemRef<uint16_t, 4> *kv1,
    MemRef<long long, 1> *pos1,
    MemRef<uint16_t, 4> *kv2,
    MemRef<uint16_t, 4> *kv3,
    MemRef<long long, 1> *pos2,
    MemRef<uint16_t, 4> *kv4,
    MemRef<uint16_t, 4> *kv5,
    MemRef<long long, 1> *pos3,
    MemRef<uint16_t, 4> *kv6,
    MemRef<uint16_t, 4> *kv7,
    MemRef<long long, 1> *pos4,
    MemRef<uint16_t, 4> *kv8,
    MemRef<uint16_t, 4> *kv9,
    MemRef<long long, 1> *pos5,
    MemRef<uint16_t, 4> *kv10,
    MemRef<uint16_t, 4> *kv11,
    MemRef<long long, 1> *pos6,
    MemRef<uint16_t, 4> *kv12,
    MemRef<uint16_t, 4> *kv13,
    MemRef<long long, 1> *pos7,
    MemRef<uint16_t, 4> *kv14,
    MemRef<uint16_t, 4> *kv15,
    MemRef<long long, 1> *pos8,
    MemRef<uint16_t, 4> *kv16,
    MemRef<uint16_t, 4> *kv17,
    MemRef<long long, 1> *pos9,
    MemRef<uint16_t, 4> *kv18,
    MemRef<uint16_t, 4> *kv19,
    MemRef<long long, 1> *pos10,
    MemRef<uint16_t, 4> *kv20,
    MemRef<uint16_t, 4> *kv21,
    MemRef<long long, 1> *pos11,
    MemRef<uint16_t, 4> *kv22,
    MemRef<uint16_t, 4> *kv23,
    MemRef<long long, 1> *pos12,
    MemRef<uint16_t, 4> *kv24,
    MemRef<uint16_t, 4> *kv25,
    MemRef<long long, 1> *pos13,
    MemRef<uint16_t, 4> *kv26,
    MemRef<uint16_t, 4> *kv27,
    MemRef<long long, 1> *pos14,
    MemRef<uint16_t, 4> *kv28,
    MemRef<uint16_t, 4> *kv29,
    MemRef<long long, 1> *pos15,
    MemRef<uint16_t, 4> *kv30,
    MemRef<uint16_t, 4> *kv31,
    MemRef<long long, 1> *pos16,
    MemRef<uint16_t, 4> *kv32,
    MemRef<uint16_t, 4> *kv33,
    MemRef<long long, 1> *pos17,
    MemRef<uint16_t, 4> *kv34,
    MemRef<uint16_t, 4> *kv35,
    MemRef<long long, 1> *pos18,
    MemRef<uint16_t, 4> *kv36,
    MemRef<uint16_t, 4> *kv37,
    MemRef<long long, 1> *pos19,
    MemRef<uint16_t, 4> *kv38,
    MemRef<uint16_t, 4> *kv39,
    MemRef<long long, 1> *pos20,
    MemRef<uint16_t, 4> *kv40,
    MemRef<uint16_t, 4> *kv41,
    MemRef<long long, 1> *pos21,
    MemRef<uint16_t, 4> *kv42,
    MemRef<uint16_t, 4> *kv43,
    MemRef<long long, 1> *pos22,
    MemRef<uint16_t, 4> *kv44,
    MemRef<uint16_t, 4> *kv45,
    MemRef<long long, 1> *pos23,
    MemRef<uint16_t, 4> *kv46,
    MemRef<uint16_t, 4> *kv47,
    MemRef<long long, 1> *pos24,
    MemRef<uint16_t, 4> *kv48,
    MemRef<uint16_t, 4> *kv49,
    MemRef<long long, 1> *pos25,
    MemRef<uint16_t, 4> *kv50,
    MemRef<uint16_t, 4> *kv51,
    MemRef<long long, 1> *pos26,
    MemRef<uint16_t, 4> *kv52,
    MemRef<uint16_t, 4> *kv53,
    MemRef<long long, 1> *pos27,
    MemRef<uint16_t, 4> *kv54,
    MemRef<uint16_t, 4> *kv55);

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

/// Capture input message.
void getUserInput(std::string &inputStr) {
  std::cout << "\nPlease send a message:" << std::endl;
  std::cout << ">>> ";
  getline(std::cin, inputStr);
  std::cout << std::endl;
}

/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

/// Print information for each iteration.
void printIterInfo(size_t iterIdx, std::string str, double time) {
  total_time += time;
  std::cout << "\033[32;1m[Iteration " << iterIdx << "] \033[0m";
  std::cout << "Token: " << str << " | "
            << "Time: " << time << "s" << std::endl;
}

/// Tokenize input data in the container.
void tokenizeInput(const std::string &vocabFile,
                   Text<size_t, 2> &inputContainer) {
  printLogLabel();
  std::cout << "Vocab file: " << std::filesystem::canonical(vocabFile)
            << std::endl;
  const auto buddyTokenizeStart = std::chrono::high_resolution_clock::now();
  inputContainer.tokenizeQwen3(vocabFile, MaxTokenLength);
  const auto buddyTokenizeEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> buddyTokenizeTime =
      buddyTokenizeEnd - buddyTokenizeStart;
  printLogLabel();
  std::cout << "Tokenize time: " << buddyTokenizeTime.count() << "ms"
            << std::endl;
}

/// Load parameters into data container.
template <typename T>
void loadParameters(const std::string &paramFilePath, MemRef<T, 1> &params) {
  const auto loadStart = std::chrono::high_resolution_clock::now();
  std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
  if (!paramFile.is_open()) {
    throw std::runtime_error("[Error] Failed to open params file!");
  }
  printLogLabel();
  std::cout << "Loading params..." << std::endl;
  printLogLabel();
  std::cout << "Params file: " << std::filesystem::canonical(paramFilePath)
            << std::endl;
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(T) * params.getSize());
  if (paramFile.fail()) {
    throw std::runtime_error("Error occurred while reading params file!");
  }
  paramFile.close();
  const auto loadEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> loadTime =
      loadEnd - loadStart;
  printLogLabel();
  std::cout << "Params load time: " << (double)(loadTime.count()) / 1000
            << "s\n"
            << std::endl;
}

/// Find the index of the max value.
static float bf16ToF32(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

int findMaxIndex(const uint16_t *start, const uint16_t *end) {
  if (start >= end) {
    throw std::runtime_error("Empty logits buffer");
  }
  int maxIdx = 0;
  float maxVal = bf16ToF32(start[0]);
  for (int i = 1; start + i < end; ++i) {
    float val = bf16ToF32(start[i]);
    if (val > maxVal) {
      maxVal = val;
      maxIdx = i;
    }
  }
  return maxIdx;
}

void copy_kv_by_cache_position_block(KVPtrArray &prefill_kvs,
                                     KVPtrArray &decode_kvs, int copy_len) {
  if (copy_len < 0 || copy_len > (int)MaxTokenLength) {
    throw std::runtime_error("KV copy_len out of range");
  }
  for (int k = 0; k < 56; ++k) {
    auto &src = *prefill_kvs[k];
    auto &dst = *decode_kvs[k];
    for (int h = 0; h < (int)HeadNum; ++h) {
      size_t bytes_to_copy =
          static_cast<size_t>(copy_len) * HiddenSize * sizeof(uint16_t);
      uint16_t *src_ptr = src.getData() + h * MaxTokenLength * HiddenSize;
      uint16_t *dst_ptr = dst.getData() + h * MaxTokenLength * HiddenSize;
      std::memcpy(dst_ptr, src_ptr, bytes_to_copy);
    }
  }
}

void sync_cache_pos(MemRefContainer &c, long long pos) {
  if (pos < 0 || pos >= (long long)MaxTokenLength) {
    throw std::runtime_error("cache_position out of KV range");
  }
  MemRef<long long, 1> *posPtrs[28] = {
      &c.pos0,  &c.pos1,  &c.pos2,  &c.pos3,  &c.pos4,  &c.pos5,  &c.pos6,
      &c.pos7,  &c.pos8,  &c.pos9,  &c.pos10, &c.pos11, &c.pos12, &c.pos13,
      &c.pos14, &c.pos15, &c.pos16, &c.pos17, &c.pos18, &c.pos19, &c.pos20,
      &c.pos21, &c.pos22, &c.pos23, &c.pos24, &c.pos25, &c.pos26, &c.pos27};
  for (int pi = 0; pi < 28; ++pi)
    posPtrs[pi]->getData()[0] = pos;
}

// -----------------------------------------------------------------------------
// Qwen3-0.6B Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  /// Print the title of this example.
  const std::string title = "Qwen3-0.6B Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  /// Define directories of vacabulary and parameter file.
  std::string qwen3_0_6b_Dir = QWEN3_0_6B_EXAMPLE_PATH;
  std::string qwen3_0_6b_BuildDir = QWEN3_0_6B_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = qwen3_0_6b_Dir + "vocab.txt";
  const std::string paramsBf16Dir = qwen3_0_6b_BuildDir + "arg0_0_6b.data";
  const std::string paramsF32Dir = qwen3_0_6b_BuildDir + "arg1_0_6b.data";

  /// Get user message.
  std::string inputStr;
  getUserInput(inputStr);

  /// Initialize data containers
  //  - Input container.
  //  - Result container
  //  - Output container.
  //  - Parameters container.
  Text<size_t, 2> outputContainer;
  Text<size_t, 2> inputContainerPrefill(inputStr);
  MemRef<long long, 2> inputContainerDecode({1, 1}, 0LL);
  MemRef<uint16_t, 1> ParamsBf16Container({ParamsBf16Size});
  MemRef<float, 1> ParamsF32Container({ParamsF32Size});
  MemRef<long long, 1> cachePosition({1}, 0LL);

  MemRef<uint16_t, 3> logits_prefill({1, MaxTokenLength, MaxVocabSize});

  MemRef<uint16_t, 4> kv0({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv1({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv2({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv3({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv4({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv5({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv6({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv7({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

  MemRef<uint16_t, 4> kv8({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv9({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv10({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv11({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv12({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv13({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv14({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv15({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

  MemRef<uint16_t, 4> kv16({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv17({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv18({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv19({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv20({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv21({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv22({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv23({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

  MemRef<uint16_t, 4> kv24({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv25({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv26({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv27({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv28({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv29({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv30({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv31({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

  MemRef<uint16_t, 4> kv32({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv33({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv34({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv35({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv36({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv37({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv38({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv39({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

  MemRef<uint16_t, 4> kv40({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv41({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv42({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv43({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv44({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv45({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv46({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv47({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

  MemRef<uint16_t, 4> kv48({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv49({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv50({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv51({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv52({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv53({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv54({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv55({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

  MemRefContainer prefillResultContainer(
      kv0, kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8, kv9, kv10, kv11, kv12, kv13,
      kv14, kv15, kv16, kv17, kv18, kv19, kv20, kv21, kv22, kv23, kv24, kv25,
      kv26, kv27, kv28, kv29, kv30, kv31, kv32, kv33, kv34, kv35, kv36, kv37,
      kv38, kv39, kv40, kv41, kv42, kv43, kv44, kv45, kv46, kv47, kv48, kv49,
      kv50, kv51, kv52, kv53, kv54, kv55, logits_prefill);
  MemRefContainer *ptrPrefillResultContainer = &prefillResultContainer;

  /// Fill data into containers
  //  - Input: register vocabulary and tokenize the input string.
  //  - Output: register vocabulary.
  //  - Parameters: load parameters from the `arg0` file into the container.
  tokenizeInput(vocabDir, inputContainerPrefill);
  outputContainer.loadVocab(vocabDir);
  loadParameters(paramsBf16Dir, ParamsBf16Container);
  loadParameters(paramsF32Dir, ParamsF32Container);

  /// Run Qwen3 Inference
  //  - Perform the forward function.
  //  - Find and append the generated token.
  //  - Continue iterating until the terminal condition is met.

  double prefillTokensPerSec = 0.0;
  const auto inferenceStart = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward_prefill(
      ptrPrefillResultContainer, &ParamsBf16Container, &ParamsF32Container,
      &inputContainerPrefill,
      &ptrPrefillResultContainer->pos0,
      &ptrPrefillResultContainer->pos1,
      &ptrPrefillResultContainer->pos2,
      &ptrPrefillResultContainer->pos3,
      &ptrPrefillResultContainer->pos4,
      &ptrPrefillResultContainer->pos5,
      &ptrPrefillResultContainer->pos6,
      &ptrPrefillResultContainer->pos7,
      &ptrPrefillResultContainer->pos8,
      &ptrPrefillResultContainer->pos9,
      &ptrPrefillResultContainer->pos10,
      &ptrPrefillResultContainer->pos11,
      &ptrPrefillResultContainer->pos12,
      &ptrPrefillResultContainer->pos13,
      &ptrPrefillResultContainer->pos14,
      &ptrPrefillResultContainer->pos15,
      &ptrPrefillResultContainer->pos16,
      &ptrPrefillResultContainer->pos17,
      &ptrPrefillResultContainer->pos18,
      &ptrPrefillResultContainer->pos19,
      &ptrPrefillResultContainer->pos20,
      &ptrPrefillResultContainer->pos21,
      &ptrPrefillResultContainer->pos22,
      &ptrPrefillResultContainer->pos23,
      &ptrPrefillResultContainer->pos24,
      &ptrPrefillResultContainer->pos25,
      &ptrPrefillResultContainer->pos26,
      &ptrPrefillResultContainer->pos27);
  const auto inferenceEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferenceTime =
      inferenceEnd - inferenceStart;

  int tokenIndex = inputContainerPrefill.getTokenCnt() - 1;
  const uint16_t *startPtr =
      ptrPrefillResultContainer->logits.getData() + tokenIndex * MaxVocabSize;
  const uint16_t *endPtr = startPtr + MaxVocabSize;
  int maxIndex = findMaxIndex(startPtr, endPtr);
  std::string tok = inputContainerPrefill.getStr(maxIndex);
  printIterInfo(0, tok, inferenceTime.count() / 1000);
  const double prefillSeconds = inferenceTime.count() / 1000.0;
  if (prefillSeconds > 0.0) {
    prefillTokensPerSec = static_cast<double>(MaxTokenLength) / prefillSeconds;
  }
  inputContainerDecode.getData()[0] = (long long)maxIndex;
  outputContainer.appendTokenIdx(maxIndex);

  MemRef<uint16_t, 3> logits_decode({1, 1, MaxVocabSize});

  MemRefContainer decodeResultContainer(
      kv0, kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8, kv9, kv10, kv11, kv12, kv13,
      kv14, kv15, kv16, kv17, kv18, kv19, kv20, kv21, kv22, kv23, kv24, kv25,
      kv26, kv27, kv28, kv29, kv30, kv31, kv32, kv33, kv34, kv35, kv36, kv37,
      kv38, kv39, kv40, kv41, kv42, kv43, kv44, kv45, kv46, kv47, kv48, kv49,
      kv50, kv51, kv52, kv53, kv54, kv55, logits_decode);

  MemRefContainer *ptrDecodeResultContainer = &decodeResultContainer;
  KVPtrArray prefillKVs = buildKVPtrs(prefillResultContainer);
  KVPtrArray decodeKVs = buildKVPtrs(decodeResultContainer);

  const size_t tokenCnt = inputContainerPrefill.getTokenCnt();
  // Match BuddyQwen3: first decode writes at tokenCnt+1. Remaining slots are
  // [tokenCnt+1, MaxTokenLength).
  long long nextPos = static_cast<long long>(tokenCnt) + 1;
  if (nextPos > (long long)MaxTokenLength) {
    throw std::runtime_error(
        "prefill tokenCnt too large for decode: tokenCnt+1 > MaxTokenLength");
  }
  int copy_len = static_cast<int>(
      std::min(static_cast<size_t>(nextPos), MaxTokenLength));
  copy_kv_by_cache_position_block(prefillKVs, decodeKVs, copy_len);

  cachePosition.getData()[0] = nextPos;
  int generateLen = static_cast<int>(MaxTokenLength) - static_cast<int>(nextPos);
  printLogLabel();
  std::cout << "tokenCnt=" << tokenCnt << " nextPos=" << nextPos
            << " generateLen=" << generateLen << std::endl;
  double decodeTimeAccumMs = 0.0;
  size_t decodeTokens = 0;
  for (int i = 1; i <= generateLen; i++) {
    sync_cache_pos(*ptrDecodeResultContainer, cachePosition.getData()[0]);
    const auto inferenceStart = std::chrono::high_resolution_clock::now();
    _mlir_ciface_forward_decode(
        ptrDecodeResultContainer, &ParamsBf16Container, &ParamsF32Container,
        &inputContainerDecode,
        &ptrDecodeResultContainer->pos0,
        &ptrDecodeResultContainer->kv0,
        &ptrDecodeResultContainer->kv1,
        &ptrDecodeResultContainer->pos1,
        &ptrDecodeResultContainer->kv2,
        &ptrDecodeResultContainer->kv3,
        &ptrDecodeResultContainer->pos2,
        &ptrDecodeResultContainer->kv4,
        &ptrDecodeResultContainer->kv5,
        &ptrDecodeResultContainer->pos3,
        &ptrDecodeResultContainer->kv6,
        &ptrDecodeResultContainer->kv7,
        &ptrDecodeResultContainer->pos4,
        &ptrDecodeResultContainer->kv8,
        &ptrDecodeResultContainer->kv9,
        &ptrDecodeResultContainer->pos5,
        &ptrDecodeResultContainer->kv10,
        &ptrDecodeResultContainer->kv11,
        &ptrDecodeResultContainer->pos6,
        &ptrDecodeResultContainer->kv12,
        &ptrDecodeResultContainer->kv13,
        &ptrDecodeResultContainer->pos7,
        &ptrDecodeResultContainer->kv14,
        &ptrDecodeResultContainer->kv15,
        &ptrDecodeResultContainer->pos8,
        &ptrDecodeResultContainer->kv16,
        &ptrDecodeResultContainer->kv17,
        &ptrDecodeResultContainer->pos9,
        &ptrDecodeResultContainer->kv18,
        &ptrDecodeResultContainer->kv19,
        &ptrDecodeResultContainer->pos10,
        &ptrDecodeResultContainer->kv20,
        &ptrDecodeResultContainer->kv21,
        &ptrDecodeResultContainer->pos11,
        &ptrDecodeResultContainer->kv22,
        &ptrDecodeResultContainer->kv23,
        &ptrDecodeResultContainer->pos12,
        &ptrDecodeResultContainer->kv24,
        &ptrDecodeResultContainer->kv25,
        &ptrDecodeResultContainer->pos13,
        &ptrDecodeResultContainer->kv26,
        &ptrDecodeResultContainer->kv27,
        &ptrDecodeResultContainer->pos14,
        &ptrDecodeResultContainer->kv28,
        &ptrDecodeResultContainer->kv29,
        &ptrDecodeResultContainer->pos15,
        &ptrDecodeResultContainer->kv30,
        &ptrDecodeResultContainer->kv31,
        &ptrDecodeResultContainer->pos16,
        &ptrDecodeResultContainer->kv32,
        &ptrDecodeResultContainer->kv33,
        &ptrDecodeResultContainer->pos17,
        &ptrDecodeResultContainer->kv34,
        &ptrDecodeResultContainer->kv35,
        &ptrDecodeResultContainer->pos18,
        &ptrDecodeResultContainer->kv36,
        &ptrDecodeResultContainer->kv37,
        &ptrDecodeResultContainer->pos19,
        &ptrDecodeResultContainer->kv38,
        &ptrDecodeResultContainer->kv39,
        &ptrDecodeResultContainer->pos20,
        &ptrDecodeResultContainer->kv40,
        &ptrDecodeResultContainer->kv41,
        &ptrDecodeResultContainer->pos21,
        &ptrDecodeResultContainer->kv42,
        &ptrDecodeResultContainer->kv43,
        &ptrDecodeResultContainer->pos22,
        &ptrDecodeResultContainer->kv44,
        &ptrDecodeResultContainer->kv45,
        &ptrDecodeResultContainer->pos23,
        &ptrDecodeResultContainer->kv46,
        &ptrDecodeResultContainer->kv47,
        &ptrDecodeResultContainer->pos24,
        &ptrDecodeResultContainer->kv48,
        &ptrDecodeResultContainer->kv49,
        &ptrDecodeResultContainer->pos25,
        &ptrDecodeResultContainer->kv50,
        &ptrDecodeResultContainer->kv51,
        &ptrDecodeResultContainer->pos26,
        &ptrDecodeResultContainer->kv52,
        &ptrDecodeResultContainer->kv53,
        &ptrDecodeResultContainer->pos27,
        &ptrDecodeResultContainer->kv54,
        &ptrDecodeResultContainer->kv55);

    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;
    decodeTimeAccumMs += inferenceTime.count();
    decodeTokens += 1;

    // Determine the generated token.
    const uint16_t *startPtr = ptrDecodeResultContainer->logits.getData();
    const uint16_t *endPtr = startPtr + MaxVocabSize;
    maxIndex = findMaxIndex(startPtr, endPtr);
    std::string tok = inputContainerPrefill.getStr(maxIndex);
    // Print the generated token and inference time.
    printIterInfo(i, tok, inferenceTime.count() / 1000);

    // Stop if a <|end▁of▁sentence|> token is generated.
    if (maxIndex == 151643) {
      break;
    }
    // Append the generated token into the input and output container.
    inputContainerDecode.getData()[0] = maxIndex;
    outputContainer.appendTokenIdx(maxIndex);
    cachePosition.getData()[0] += 1;
  }

  const double decodeSeconds = decodeTimeAccumMs / 1000.0;
  const double decodeTokensPerSec =
      decodeSeconds > 0.0 ? static_cast<double>(decodeTokens) / decodeSeconds
                          : 0.0;

  /// Print the final result
  std::cout << "\n\033[33;1m[Total time]\033[0m " << total_time << std::endl;
  std::cout << "\033[33;1m[Prefilling]\033[0m " << prefillTokensPerSec
            << " tokens/s" << std::endl;
  std::cout << "\033[33;1m[Decoding]\033[0m " << decodeTokensPerSec
            << " tokens/s" << std::endl;
  std::cout << "\033[33;1m[Input]\033[0m " << inputStr << std::endl;
  std::cout << "\033[33;1m[Output]\033[0m " << outputContainer.revertQwen3()
            << std::endl;

  return 0;
}

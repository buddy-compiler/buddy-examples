// The GPU side of GVN for pi0-FAST on piplus: runs its device programs
// (their launches calls of the linalg graphs, gvn-gpu-program) on the GPU as
// baremetal.cpp runs them on the device: the prefill (step 0) on the
// cameras and the prompt, then the decoding step once per token (steps 1,
// 2, ...) until the EOS, keeping between the steps the keys and values of the
// generated tokens and the bias that sees them; the tokens are checked
// against those of the quantized program in PyTorch (generated.data), as
// baremetal.cpp reports them. The values the slices read and write are
// recorded as the programs compute them: gvn_launch names the launch,
// gvn_record_<type> a value (kind 0: an operand of the launch, 1: a value of
// its graph, 2: a result), each written to the values file (bytes equal to a
// value already there are not written again, the line names that offset),
// one line of the index each:
//   step launch kind index element-bytes rank sizes... offset bytes
//
// gpu <program manifest> <values file> <index file> <model dir> <harness dir>
//
// The program manifest (gvn-gpu-program) gives for each program its name,
// argument count and count of packs, the ranks of its results, then each
// argument (element bytes, rank, sizes); packs.txt of the harness dir names per
// program its parameter packs, files of ddr/ of the model dir.
#include "model.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" {
// The CUDA driver: the GPU computes asynchronously; a recorded value is
// copied once every launch before it ended. The program runs in the primary
// context of device 0 (mlir_cuda_runtime).
int cuInit(unsigned);
int cuDeviceGet(int *, int);
int cuDevicePrimaryCtxRetain(void **, int);
int cuCtxPushCurrent_v2(void *);
int cuCtxPopCurrent_v2(void **);
int cuCtxSynchronize(void);
int cuMemcpy(unsigned long long, unsigned long long, size_t);
int cuMemAllocManaged(unsigned long long *, size_t, unsigned);
}

namespace {
struct Descriptor {
  char *allocated;
  char *aligned;
  int64_t offset;
  int64_t rest[];
};

struct Unranked {
  int64_t rank;
  Descriptor *descriptor;
};

FILE *values, *lines;
int64_t step, launch;
uint64_t written;
// The values written: the hash of their bytes -> (offset, bytes).
std::unordered_multimap<size_t, std::pair<uint64_t, size_t>> stored;
void *context;

void check(int status, const char *what) {
  if (status) {
    std::fprintf(stderr, "%s failed: %d\n", what, status);
    std::exit(1);
  }
}

// A copy of `bytes` in managed memory, which the GPU and the host both read:
// the program's buffers all are (gvn-managed-memory).
char *managed(const void *bytes, size_t count) {
  unsigned long long pointer;
  check(cuCtxPushCurrent_v2(context), "cuCtxPushCurrent");
  check(cuMemAllocManaged(&pointer, count ? count : 1, 1), "cuMemAllocManaged");
  void *popped;
  check(cuCtxPopCurrent_v2(&popped), "cuCtxPopCurrent");
  std::memcpy((void *)pointer, bytes, count);
  return (char *)pointer;
}

std::vector<char> load(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::fprintf(stderr, "cannot read %s\n", path.c_str());
    std::exit(1);
  }
  std::vector<char> bytes(file.tellg());
  file.seekg(0);
  file.read(bytes.data(), bytes.size());
  return bytes;
}

void record(int64_t kind, int64_t position, int64_t element, Unranked *value) {
  const int64_t rank = value->rank;
  const Descriptor *descriptor = value->descriptor;
  const int64_t *sizes = descriptor->rest, *strides = descriptor->rest + rank;
  int64_t count = 1, span = 1;
  for (int64_t dim = 0; dim < rank; ++dim) {
    count *= sizes[dim];
    if (sizes[dim])
      span += (sizes[dim] - 1) * strides[dim];
  }
  std::vector<char> host(count * element);
  if (count) {
    std::vector<char> region(span * element);
    check(cuCtxPushCurrent_v2(context), "cuCtxPushCurrent");
    check(cuCtxSynchronize(), "cuCtxSynchronize");
    check(cuMemcpy((unsigned long long)region.data(),
                   (unsigned long long)(descriptor->aligned +
                                        descriptor->offset * element),
                   region.size()),
          "cuMemcpy");
    void *popped;
    check(cuCtxPopCurrent_v2(&popped), "cuCtxPopCurrent");
    // Gather in row-major order.
    std::vector<int64_t> at(rank, 0);
    for (int64_t i = 0; i < count; ++i) {
      int64_t source = 0;
      for (int64_t dim = 0; dim < rank; ++dim)
        source += at[dim] * strides[dim];
      std::memcpy(&host[i * element], &region[source * element], element);
      for (int64_t dim = rank - 1; dim >= 0 && ++at[dim] == sizes[dim]; --dim)
        at[dim] = 0;
    }
  }
  // The bytes are written once: a value the file already holds (a weight
  // view every step records again) names that offset.
  const std::string_view bytes(host.data(), host.size());
  uint64_t offset = written;
  auto [first, last] = stored.equal_range(std::hash<std::string_view>{}(bytes));
  for (auto each = first; each != last && offset == written; ++each) {
    if (each->second.second != host.size())
      continue;
    std::vector<char> held(host.size());
    std::fflush(values);
    if (pread(fileno(values), held.data(), held.size(), each->second.first) !=
        (ssize_t)held.size()) {
      std::fprintf(stderr, "cannot read back the values file\n");
      std::exit(1);
    }
    if (std::memcmp(held.data(), host.data(), host.size()) == 0)
      offset = each->second.first;
  }
  if (offset == written) {
    std::fwrite(host.data(), 1, host.size(), values);
    stored.emplace(std::hash<std::string_view>{}(bytes),
                   std::pair<uint64_t, size_t>{written, host.size()});
    written += host.size();
  }
  std::fprintf(lines, "%ld %ld %ld %ld %ld %ld", step, launch, kind, position,
               element, rank);
  for (int64_t dim = 0; dim < rank; ++dim)
    std::fprintf(lines, " %ld", sizes[dim]);
  std::fprintf(lines, " %lu %zu\n", offset, host.size());
}
} // namespace

extern "C" {
void _mlir_ciface_gvn_launch(int64_t number) { launch = number; }
void _mlir_ciface_gvn_record_f32(int64_t kind, int64_t position,
                                 Unranked *value) {
  record(kind, position, 4, value);
}
void _mlir_ciface_gvn_record_i1(int64_t kind, int64_t position,
                                Unranked *value) {
  record(kind, position, 1, value);
}
void _mlir_ciface_gvn_record_i32(int64_t kind, int64_t position,
                                 Unranked *value) {
  record(kind, position, 4, value);
}
void _mlir_ciface_gvn_record_i8(int64_t kind, int64_t position,
                                Unranked *value) {
  record(kind, position, 1, value);
}
void _mlir_ciface_gvn_record_i64(int64_t kind, int64_t position,
                                 Unranked *value) {
  record(kind, position, 8, value);
}
}

// The programs (gvn-packed-entry): the pointer to their results, then the
// pointers to the descriptors of their arguments.
extern "C" void gvn_prefill_program(void *results, void **arguments);
extern "C" void gvn_decode_program(void *results, void **arguments);

namespace {
// A descriptor of `rank` dimensions, contiguous, at `data`.
std::vector<int64_t> describe(char *data, const std::vector<int64_t> &sizes) {
  const int64_t rank = sizes.size();
  std::vector<int64_t> words(3 + 2 * rank);
  words[0] = words[1] = (int64_t)data;
  int64_t stride = 1;
  for (int64_t dim = rank - 1; dim >= 0; --dim) {
    words[3 + dim] = sizes[dim];
    words[3 + rank + dim] = stride;
    stride *= sizes[dim];
  }
  return words;
}

// A program of the manifest: its arguments' shapes, its packs (their data in
// managed memory) and its results' ranks.
struct Program {
  std::vector<std::vector<int64_t>> shapes;
  std::vector<char *> packs;
  std::vector<int64_t> ranks;
};

// Calls `entry` on `data` (one pointer an argument, of the shapes of the
// program); the results, a descriptor each, their data in managed memory.
std::vector<int64_t *> call(void (*entry)(void *, void **),
                            const Program &program,
                            const std::vector<char *> &data,
                            std::vector<int64_t> &results) {
  std::vector<std::vector<int64_t>> descriptors(data.size());
  std::vector<void *> pointers(data.size());
  for (size_t each = 0; each < data.size(); ++each) {
    descriptors[each] = describe(data[each], program.shapes[each]);
    pointers[each] = descriptors[each].data();
  }
  size_t words = 0;
  for (int64_t rank : program.ranks)
    words += 3 + 2 * rank;
  results.assign(words, 0);
  entry(results.data(), pointers.data());
  check(cuCtxPushCurrent_v2(context), "cuCtxPushCurrent");
  check(cuCtxSynchronize(), "cuCtxSynchronize");
  void *popped;
  check(cuCtxPopCurrent_v2(&popped), "cuCtxPopCurrent");
  std::vector<int64_t *> descriptorsOf;
  size_t at = 0;
  for (int64_t rank : program.ranks) {
    descriptorsOf.push_back(&results[at]);
    at += 3 + 2 * rank;
  }
  return descriptorsOf;
}

// The data of a result descriptor.
template <typename T> T *dataOf(int64_t *descriptor) {
  return (T *)descriptor[1] + descriptor[2];
}

// The most likely token of the logits.
int64_t likeliest(const float *logits) {
  int64_t best = 0;
  for (int64_t i = 1; i < VOCABULARY; ++i)
    if (logits[i] > logits[best])
      best = i;
  return best;
}
} // namespace

int main(int argc, char **argv) {
  values = std::fopen(argv[2], "w+b");
  lines = std::fopen(argv[3], "w");
  const std::string model = argv[4];
  int device;
  check(cuInit(0), "cuInit");
  check(cuDeviceGet(&device, 0), "cuDeviceGet");
  check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");

  // The programs: element bytes, rank, sizes of each argument; their packs.
  std::ifstream manifest(argv[1]);
  std::ifstream packList(std::string(argv[5]) + "/packs.txt");
  std::unordered_map<std::string, Program> programs;
  for (std::string name; manifest >> name;) {
    size_t count, packs;
    manifest >> count >> packs;
    Program &program = programs[name];
    std::string line;
    std::getline(manifest, line);
    std::getline(manifest, line);
    std::istringstream ranks(line);
    for (int64_t rank; ranks >> rank;)
      program.ranks.push_back(rank);
    program.shapes.resize(count);
    for (auto &shape : program.shapes) {
      int64_t element, rank;
      manifest >> element >> rank;
      shape.resize(rank);
      for (int64_t &size : shape)
        manifest >> size;
    }
    std::string listed;
    packList >> listed;
    std::getline(packList, line);
    std::istringstream files(line);
    for (std::string file; files >> file;) {
      std::vector<char> bytes = load(model + "/ddr/" + file);
      program.packs.push_back(managed(bytes.data(), bytes.size()));
    }
    if (listed != name || program.packs.size() != packs) {
      std::fprintf(stderr,
                   "packs.txt names %zu packs of %s, the manifest %zu of %s\n",
                   program.packs.size(), listed.c_str(), packs, name.c_str());
      return 1;
    }
  }
  const Program &prefill = programs.at("prefill_program");
  const Program &decode = programs.at("decode_program");

  std::vector<char> observationFile = load(model + "/observation.data");
  const auto &observation = *(const Observation *)observationFile.data();
  std::vector<char> tableFile = load(model + "/ddr/embedding.data");
  const float *table = (const float *)tableFile.data();

  // The prefill: its packs, the images, the rows of the prompt and its mask;
  // the logits of the first token, the position of the next, the bias, the
  // keys and values of every layer.
  std::vector<float> prompt(201 * HIDDEN);
  for (unsigned i = 0; i < 201; ++i)
    std::memcpy(&prompt[i * HIDDEN], table + observation.tokens[i] * HIDDEN,
                HIDDEN * sizeof(float));
  std::vector<char *> data(prefill.packs);
  data.push_back(managed(observation.images, sizeof(observation.images)));
  data.push_back(managed(prompt.data(), prompt.size() * sizeof(float)));
  data.push_back(managed(observation.masks, sizeof(observation.masks)));
  std::vector<int64_t> prefilled;
  std::vector<int64_t *> outputs =
      call(gvn_prefill_program, prefill, data, prefilled);
  std::printf("Captured step 0\n");
  std::fflush(stdout);
  const int64_t position = *dataOf<int64_t>(outputs[1]);
  float *bias = dataOf<float>(outputs[2]);
  std::vector<int64_t> generated{likeliest(dataOf<float>(outputs[0]))};

  // A decoding step: its packs, the row of the token before, its position,
  // the bias, the keys and values of the prefill and those of the generated
  // tokens; the logits of the next token and its own keys and values, which
  // go after the generated ones, the bias seeing them from the next step on.
  std::vector<char *> before;
  for (unsigned i = 0; i < 2 * LAYERS; ++i) {
    std::vector<char> zeros(GENERATED * HEAD_DIM * sizeof(float));
    before.push_back(managed(zeros.data(), zeros.size()));
  }
  char *row = managed(table, HIDDEN * sizeof(float));
  int64_t place = 0;
  char *placeData = managed(&place, sizeof(place));
  std::vector<int64_t> decoded;
  for (unsigned token = 0; token < GENERATED - 1 && generated.back() != EOS;
       ++token) {
    step = token + 1;
    std::memcpy(row, table + generated.back() * HIDDEN, HIDDEN * sizeof(float));
    place = position + token;
    std::memcpy(placeData, &place, sizeof(place));
    data.assign(decode.packs.begin(), decode.packs.end());
    data.push_back(row);
    data.push_back(placeData);
    data.push_back((char *)bias);
    for (unsigned i = 0; i < 2 * LAYERS; ++i)
      data.push_back((char *)dataOf<float>(outputs[3 + i]));
    for (char *each : before)
      data.push_back(each);
    std::vector<int64_t *> results =
        call(gvn_decode_program, decode, data, decoded);
    std::printf("Captured step %u\n", token + 1);
    std::fflush(stdout);
    for (unsigned i = 0; i < 2 * LAYERS; ++i)
      std::memcpy(before[i] + token * HEAD_DIM * sizeof(float),
                  dataOf<float>(results[1 + i]), HEAD_DIM * sizeof(float));
    bias[PREFIX_KEYS + token] = 0.0f;
    generated.push_back(likeliest(dataOf<float>(results[0])));
  }
  std::fclose(values);
  std::fclose(lines);

  // The tokens against PyTorch (generated.data): its tokens up to the EOS.
  std::vector<char> expectedFile = load(model + "/generated.data");
  const auto *expected = (const int64_t *)expectedFile.data();
  unsigned different = 0, length = 0;
  while (length < GENERATED && expected[length] != 0)
    ++length;
  for (unsigned i = 0; i < generated.size() && i < length; ++i)
    different += generated[i] != expected[i];
  std::printf(
      "tokens: %zu generated, %u in PyTorch, %u of the first %zu differ\n",
      generated.size(), length, different,
      generated.size() < length ? generated.size() : size_t(length));
  return 0;
}

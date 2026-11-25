// The GPU side of GVN for the laya device program: runs the device program
// (its launches calls of the linalg graphs, gvn-gpu-program) on the GPU once,
// as the recipe's baremetal program runs it on the inputs of the prompt. The
// values the slices read and write are recorded as the program computes them:
// gvn_launch names the launch, gvn_record_<type> a value (kind 0: an operand
// of the launch, 1: a value of its graph, 2: a result), each written to the
// values file (bytes equal to a value already there are not written again,
// the line names that offset), one line of the index each:
//   step launch kind index element-bytes rank sizes... offset bytes
//
// gpu <program manifest> <values file> <index file> <model dir> <harness dir>
//
// The prompt is ddr/ of the model dir.
//
// The program manifest (gvn-gpu-program) gives the count of the program's
// arguments and of its parameter packs, the ranks of its results, then each
// argument (element bytes, rank, sizes, -1 dynamic); sizes.txt of the prompt
// gives the dynamic sizes of its inputs.
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
  // view recorded again) names that offset.
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

// The program (gvn-packed-entry): the pointer to its results, then the
// pointers to the descriptors of its arguments.
extern "C" void gvn_program(void *results, void **arguments);

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

int main(int argc, char **argv) {
  const std::string prompt = std::string(argv[4]) + "/ddr";
  values = std::fopen(argv[2], "w+b");
  lines = std::fopen(argv[3], "w");
  int device;
  check(cuInit(0), "cuInit");
  check(cuDeviceGet(&device, 0), "cuDeviceGet");
  check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");

  // The arguments of the program: element bytes, rank, sizes (-1 dynamic).
  std::ifstream manifest(argv[1]);
  std::string name;
  size_t count, packCount;
  manifest >> name >> count >> packCount;
  std::string rankLine;
  std::getline(manifest, rankLine);
  std::getline(manifest, rankLine);
  std::istringstream rankWords(rankLine);
  size_t resultWords = 0;
  for (int64_t rank; rankWords >> rank;)
    resultWords += 3 + 2 * rank;
  std::vector<int64_t> elements(count);
  std::vector<std::vector<int64_t>> shapes(count);
  for (size_t argument = 0; argument < count; ++argument) {
    int64_t rank;
    manifest >> elements[argument] >> rank;
    shapes[argument].resize(rank);
    for (int64_t &size : shapes[argument])
      manifest >> size;
  }

  // The inputs as baremetal.cpp lays them out: the packs (F32,
  // INT8, then INT32 ones), then the inputs of sizes.txt from inputs.data,
  // each 64-byte aligned.
  const char *packNames[] = {"/pack-f32.data", "/pack-i8.data",
                             "/pack-i32.data"};
  std::vector<std::vector<char>> packs(packCount);
  std::vector<char> inputs = load(prompt + "/inputs.data");
  std::vector<std::vector<int64_t>> sizes(count);
  std::vector<char *> data(count);
  for (size_t pack = 0; pack < packCount; ++pack) {
    packs[pack] = load(prompt + packNames[pack]);
    data[pack] = managed(packs[pack].data(), packs[pack].size());
    sizes[pack] = shapes[pack];
  }
  char *inputBytes = managed(inputs.data(), inputs.size());
  std::ifstream sizesFile(prompt + "/sizes.txt");
  std::string line;
  size_t argument = packCount;
  uint64_t offset = 0;
  while (std::getline(sizesFile, line)) {
    std::istringstream words(line);
    std::string kind;
    uint64_t bytes;
    words >> kind >> bytes;
    if (kind != "input")
      continue;
    sizes[argument] = shapes[argument];
    for (int64_t &size : sizes[argument])
      if (size < 0)
        words >> size;
    data[argument] = inputBytes + offset;
    offset += (bytes + 63) / 64 * 64;
    ++argument;
  }
  if (argument != count) {
    std::fprintf(stderr, "sizes.txt has %zu inputs, the program %zu\n",
                 argument - packCount, count - packCount);
    return 1;
  }
  std::vector<std::vector<int64_t>> descriptors(count);
  std::vector<void *> pointers(count);
  for (size_t each = 0; each < count; ++each) {
    descriptors[each] = describe(data[each], sizes[each]);
    pointers[each] = descriptors[each].data();
  }
  std::vector<int64_t> results(resultWords);
  gvn_program(results.data(), pointers.data());
  std::printf("Captured step 1\n");
  std::fclose(values);
  std::fclose(lines);

  // As baremetal.cpp: the valid choices and the actions within 1e-2 of the
  // quantized model in PyTorch (expected.data: the 16 choices, then the 2
  // actions), relative to its largest magnitude (check_output), and the same
  // choice and action.
  check(cuCtxPushCurrent_v2(context), "cuCtxPushCurrent");
  check(cuCtxSynchronize(), "cuCtxSynchronize");
  void *popped;
  check(cuCtxPopCurrent_v2(&popped), "cuCtxPopCurrent");
  // Results 0 and 1, rank 2 each: the choices, the actions.
  const float *choices = (const float *)results[1] + results[2];
  const float *actions = (const float *)results[7 + 1] + results[7 + 2];
  std::vector<char> golden = load(prompt + "/../expected.data");
  const float *expected = (const float *)golden.data();
  const bool *valid = (const bool *)data[packCount + 3];
  float worst = 0, scale = 0;
  unsigned choice = 0, gold = 0;
  for (unsigned i = 0; i < 18; ++i) {
    const float actual = i < 16 ? choices[i] : actions[i - 16];
    if (i < 16 && actual > choices[choice])
      choice = i;
    if (i < 16 && expected[i] > expected[gold])
      gold = i;
    if (i < 16 && !valid[i])
      continue;
    const float difference =
        actual > expected[i] ? actual - expected[i] : expected[i] - actual;
    const float magnitude = expected[i] < 0 ? -expected[i] : expected[i];
    worst = difference > worst ? difference : worst;
    scale = magnitude > scale ? magnitude : scale;
  }
  const unsigned action = actions[1] > actions[0];
  const unsigned goldAction = expected[17] > expected[16];
  std::printf("Choice: %u, reference %u\nAction: %u, reference %u\n"
              "Output max diff: %g (relative %g)\n",
              choice, gold, action, goldAction, worst, worst / scale);
  return !(worst / scale <= 1e-2f) || choice != gold || action != goldAction;
}

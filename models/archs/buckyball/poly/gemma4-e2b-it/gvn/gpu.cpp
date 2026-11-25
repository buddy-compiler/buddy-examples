// The GPU side of GVN for poly device programs: runs the device program (its
// launches calls of the linalg graphs, gvn-gpu-program) on the GPU for the
// eight steps of the recipe's baremetal program: the prompt, then
// one token a step, greedily, each step's caches the next step's inputs. The
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
  // As the recipe's baremetal program: eight tokens.
  const int steps = 8;
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
  std::vector<int64_t> ranks;
  size_t resultWords = 0;
  for (int64_t rank; rankWords >> rank;) {
    ranks.push_back(rank);
    resultWords += 3 + 2 * rank;
  }
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
  std::vector<unsigned> order;
  std::ifstream file(prompt + "/input-order.txt");
  if (!file) {
    std::fprintf(stderr, "decoder input-order.txt is missing\n");
    return 1;
  }
  unsigned value;
  while (file >> value)
    order.push_back(value);

  int64_t position = 0, next = 0;
  for (step = 0; step < steps; ++step) {
    std::vector<std::vector<int64_t>> descriptors(count);
    std::vector<void *> pointers(count);
    for (size_t each = 0; each < count; ++each) {
      descriptors[each] = describe(data[each], sizes[each]);
      pointers[each] = descriptors[each].data();
    }
    // The results: a descriptor of each, 3 + 2 * rank words.
    std::vector<int64_t> results(resultWords);
    gvn_program(results.data(), pointers.data());
    std::printf("Captured step %ld\n", step + 1);
    std::fflush(stdout);
    if (step + 1 == steps)
      break;

    // Results 0 and 1: the candidates of the next token (values, indices).
    auto result = [&](size_t which, int64_t rank) {
      int64_t *word = results.data();
      for (size_t each = 0; each < which; ++each)
        word += 3 + 2 * ranks[each];
      return std::pair<int64_t *, int64_t>{word, rank};
    };
    auto [valuesWords, valuesRank] = result(0, ranks[0]);
    auto [indicesWords, indicesRank] = result(1, ranks[1]);
    const float *candidates = (const float *)valuesWords[1] + valuesWords[2];
    const int64_t *candidateIndices =
        (const int64_t *)indicesWords[1] + indicesWords[2];
    int64_t candidateCount = 1;
    for (int64_t dim = 0; dim < valuesRank; ++dim)
      candidateCount *= valuesWords[3 + dim];
    int64_t best = 0;
    for (int64_t i = 1; i < candidateCount; ++i)
      if (candidates[i] > candidates[best])
        best = i;
    next = candidateIndices[best];
    std::printf("Token %ld: %ld\n", step + 1, next);
    std::fflush(stdout);
    position += sizes[packCount][1];
    for (size_t input = 0; input + packCount < count; ++input) {
      const size_t at = input + packCount;
      if (order[input] == 0) {
        data[at] = managed(&next, sizeof(next));
        sizes[at] = {1, 1};
      } else if (order[input] == 1) {
        data[at] = managed(&position, sizeof(position));
      } else {
        auto [words, rank] = result(order[input], ranks[order[input]]);
        data[at] = (char *)words[1] + words[2] * elements[at];
        for (int64_t dim = 0; dim < rank; ++dim)
          sizes[at][dim] = words[3 + dim];
      }
    }
  }
  std::fclose(values);
  std::fclose(lines);
  return 0;
}

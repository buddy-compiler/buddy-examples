// The GVN GPU side of the piplus device program: runs the device program
// (its launches calls of the linalg graphs, gvn-gpu-program) on the GPU once,
// for one action chunk. The values the slices read and write are recorded as
// the program computes them: gvn_launch names the launch, gvn_record_<type>
// a value (kind 0: an operand of the launch, 1: a value of its graph, 2: a
// result), each written to the values file, one line of the index each:
//   step launch kind index element-bytes rank sizes... offset bytes
//
// gpu <program manifest> <values file> <index file> <model dir> <harness dir>
//
// The program manifest (gvn-gpu-program) gives the count of the program's
// arguments and of its parameter packs, the ranks of its results, then each
// argument (element bytes, rank, sizes). The arguments are as baremetal.cpp
// passes them: the parameter packs (the files of ddr/ of the model dir that
// packs.txt of the harness dir names), then the fields of observation.data in
// order.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
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
  std::fwrite(host.data(), 1, host.size(), values);
  std::fprintf(lines, "%ld %ld %ld %ld %ld %ld", step, launch, kind, position,
               element, rank);
  for (int64_t dim = 0; dim < rank; ++dim)
    std::fprintf(lines, " %ld", sizes[dim]);
  std::fprintf(lines, " %lu %zu\n", written, host.size());
  written += host.size();
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
  values = std::fopen(argv[2], "wb");
  lines = std::fopen(argv[3], "w");
  int device;
  check(cuInit(0), "cuInit");
  check(cuDeviceGet(&device, 0), "cuDeviceGet");
  check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");

  // The arguments of the program: element bytes, rank, sizes.
  std::ifstream manifest(argv[1]);
  std::string name;
  size_t count, packs;
  manifest >> name >> count >> packs;
  // The results: a descriptor of each, 3 + 2 * rank words.
  std::string line;
  std::getline(manifest, line);
  std::getline(manifest, line);
  std::istringstream ranks(line);
  size_t words = 0;
  for (int64_t rank; ranks >> rank;)
    words += 3 + 2 * rank;
  const std::string model = argv[4];
  std::ifstream packList(std::string(argv[5]) + "/packs.txt");
  std::string program;
  packList >> program;
  std::vector<std::string> packFiles;
  std::getline(packList, line);
  std::istringstream files(line);
  for (std::string file; files >> file;)
    packFiles.push_back(file);
  if (program != name || packs != packFiles.size()) {
    std::fprintf(stderr,
                 "packs.txt names %zu packs of %s, the program %zu of %s\n",
                 packFiles.size(), program.c_str(), packs, name.c_str());
    return 1;
  }
  std::vector<char> observation = load(model + "/observation.data");
  char *observationBytes = managed(observation.data(), observation.size());
  size_t field = 0;
  std::vector<std::vector<int64_t>> descriptors(count);
  std::vector<void *> pointers(count);
  for (size_t argument = 0; argument < count; ++argument) {
    int64_t element, rank;
    manifest >> element >> rank;
    std::vector<int64_t> shape(rank);
    int64_t bytes = element;
    for (int64_t &size : shape) {
      manifest >> size;
      bytes *= size;
    }
    char *data;
    if (argument < packs) {
      const std::string path = model + "/ddr/" + packFiles[argument];
      std::vector<char> file = load(path);
      if (int64_t(file.size()) != bytes) {
        std::fprintf(stderr, "%s has %zu bytes, argument %zu %ld\n",
                     path.c_str(), file.size(), argument, bytes);
        return 1;
      }
      data = managed(file.data(), file.size());
    } else {
      data = observationBytes + field;
      field += bytes;
    }
    descriptors[argument] = describe(data, shape);
    pointers[argument] = descriptors[argument].data();
  }
  if (field != observation.size()) {
    std::fprintf(stderr,
                 "observation.data has %zu bytes, the program reads %zu\n",
                 observation.size(), field);
    return 1;
  }
  std::vector<int64_t> results(words);
  gvn_program(results.data(), pointers.data());
  std::printf("Captured the program\n");
  std::fclose(values);
  std::fclose(lines);
  return 0;
}

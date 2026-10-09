#include <array>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <functional>
#include <mutex>
#include <memory>
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <sys/resource.h>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
constexpr uint64_t maxBytes = 256ULL * 1024 * 1024;
constexpr char magic[] = "BBMUX1\n";
std::atomic<bool> stopping{false};
std::mutex outputMutex;

std::runtime_error error(const char *operation) {
  return std::runtime_error(std::string(operation) + ": " + strerror(errno));
}
void nonblocking(int fd) {
  int flags = fcntl(fd, F_GETFL);
  if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0)
    throw error("fcntl");
}
// A clean EOF is allowed only before the first byte of a new host header.
void transfer(int fd, void *buffer, size_t bytes, bool writing) {
  auto *data = static_cast<unsigned char *>(buffer);
  size_t offset = 0;
  while (offset != bytes) {
    if (stopping.load()) throw std::runtime_error("session stopped");
    ssize_t n = writing ? write(fd, data + offset, bytes - offset)
                        : read(fd, data + offset, bytes - offset);
    if (n > 0) { offset += static_cast<size_t>(n); continue; }
    if (n == 0) throw std::runtime_error(offset ? "partial frame EOF" : "EOF");
    if (errno == EINTR) continue;
    if (errno != EAGAIN && errno != EWOULDBLOCK) throw error("stream IO");
    pollfd p{fd, static_cast<short>(writing ? POLLOUT : POLLIN), 0};
    if (poll(&p, 1, 100) < 0 && errno != EINTR) throw error("poll");
  }
}
uint64_t decode(const unsigned char *p) {
  uint64_t value = 0;
  for (unsigned i = 0; i != 8; ++i) value |= uint64_t(p[i]) << (8 * i);
  return value;
}
void response(uint64_t rank, uint64_t sequence, uint64_t status,
              std::vector<unsigned char> &payload,
              const std::function<void()> &delivered = {}) {
  std::array<unsigned char, 32> header{};
  const uint64_t values[]{rank, sequence, status, payload.size()};
  for (unsigned field = 0; field != 4; ++field)
    for (unsigned byte = 0; byte != 8; ++byte)
      header[field * 8 + byte] = values[field] >> (8 * byte);
  std::lock_guard<std::mutex> lock(outputMutex);
  transfer(STDOUT_FILENO, header.data(), header.size(), true);
  transfer(STDOUT_FILENO, payload.data(), payload.size(), true);
  if (delivered) delivered();
}
void failure(uint64_t rank, uint64_t sequence, const std::string &message) {
  // All diagnostics here are ASCII, hence valid UTF-8.
  std::vector<unsigned char> payload(message.begin(), message.begin() +
                                   std::min<size_t>(4096, message.size()));
  response(rank, sequence, 1, payload);
}
struct Terminal {
  termios saved{};
  bool active = false;
  Terminal() {
    if (tcgetattr(STDIN_FILENO, &saved) == 0) {
      termios raw = saved;
      cfmakeraw(&raw);
      if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) throw error("tcsetattr");
      active = true;
    } else if (errno != ENOTTY) throw error("tcgetattr");
  }
  ~Terminal() { if (active) tcsetattr(STDIN_FILENO, TCSANOW, &saved); }
};
struct Worker {
  pid_t pid = -1;
  int input = -1, output = -1;
  std::mutex mutex;
  std::condition_variable wake;
  bool busy = false, available = false, responding = false;
  uint64_t sequence = 0, resultBytes = 0;
  std::vector<unsigned char> request;
  std::thread thread;
};
void run(Worker &w, uint64_t rank) {
  while (!stopping.load()) {
    std::unique_lock<std::mutex> lock(w.mutex);
    w.wake.wait_for(lock, std::chrono::milliseconds(100), [&] {
      return w.available || stopping.load();
    });
    if (stopping.load()) break;
    if (!w.available) continue;
    auto request = std::move(w.request);
    const uint64_t sequence = w.sequence, bytes = w.resultBytes;
    w.available = false;
    lock.unlock();
    try {
      transfer(w.input, request.data(), request.size(), true);
      std::vector<unsigned char> result(bytes);
      transfer(w.output, result.data(), result.size(), false);
      // Never expose a partial child response on the multiplexed stream.
      lock.lock();
      w.responding = true;
      w.busy = false;
      lock.unlock();
      response(rank, sequence, 0, result, [&] {
        std::lock_guard<std::mutex> deliveredLock(w.mutex);
        w.responding = false;
      });
    } catch (const std::exception &e) {
      if (!stopping.load()) {
        try { failure(rank, sequence, e.what()); } catch (...) {}
        stopping.store(true);
      }
      break;
    }
  }
}
void spawn(Worker &w, const std::string &program, const std::string &model,
           unsigned rank, unsigned cpu, const std::filesystem::path &directory) {
  int input[2], output[2], startup[2];
  if (pipe2(input, O_CLOEXEC) || pipe2(output, O_CLOEXEC) ||
      pipe2(startup, O_CLOEXEC)) throw error("pipe2");
  w.pid = fork();
  if (w.pid < 0) throw error("fork");
  if (w.pid == 0) {
    close(startup[0]);
    auto fail = [&] {
      int code = errno;
      (void)write(startup[1], &code, sizeof(code));
      _exit(127);
    };
    if (setpgid(0, 0) < 0) fail();
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(cpu, &affinity);
    if (sched_setaffinity(0, sizeof(affinity), &affinity) < 0) fail();
    if (chdir(directory.c_str()) < 0) fail();
    int diagnostics = open("stderr.log", O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (diagnostics < 0 || dup2(input[0], STDIN_FILENO) < 0 ||
        dup2(output[1], STDOUT_FILENO) < 0 || dup2(diagnostics, STDERR_FILENO) < 0)
      fail();
    const rlimit lockedMemory{RLIM_INFINITY, RLIM_INFINITY};
    if (setrlimit(RLIMIT_MEMLOCK, &lockedMemory) < 0) fail();
    const std::string index = std::to_string(rank);
    execl(program.c_str(), program.c_str(), model.c_str(), index.c_str(), nullptr);
    fail();
  }
  close(input[0]); close(output[1]); close(startup[1]);
  w.input = input[1]; w.output = output[0];
  int code = 0;
  ssize_t n;
  do { n = read(startup[0], &code, sizeof(code)); } while (n < 0 && errno == EINTR);
  close(startup[0]);
  if (n != 0) {
    if (n < 0) throw error("worker startup");
    throw std::runtime_error("worker " + std::to_string(rank) +
                             " startup: " + strerror(code));
  }
  nonblocking(w.input); nonblocking(w.output);
}
void cleanup(std::vector<std::unique_ptr<Worker>> &workers) {
  stopping.store(true);
  for (auto &w : workers) {
    w->wake.notify_all();
    if (w->pid > 0) { kill(-w->pid, SIGTERM); kill(w->pid, SIGTERM); }
  }
  for (auto &w : workers) if (w->thread.joinable()) w->thread.join();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  for (auto &w : workers) {
    if (w->pid > 0) {
      int status;
      pid_t done;
      do {
        done = waitpid(w->pid, &status, WNOHANG);
        if (done != 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      } while (std::chrono::steady_clock::now() < deadline);
      if (done == 0) {
        kill(-w->pid, SIGKILL); kill(w->pid, SIGKILL);
        do { done = waitpid(w->pid, &status, 0); } while (done < 0 && errno == EINTR);
      }
    }
    if (w->input >= 0) close(w->input);
    if (w->output >= 0) close(w->output);
  }
}
} // namespace

int main(int argc, char **argv) {
  std::vector<std::unique_ptr<Worker>> workers;
  std::unique_ptr<Terminal> terminal;
  int status = 1;
  bool cleaned = false;
  signal(SIGPIPE, SIG_IGN);
  // Bootstrap may redirect fd 2 to the same console as fd 1.
  int diagnostics = open("worker-mux.stderr.log", O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (diagnostics < 0 || dup2(diagnostics, STDERR_FILENO) < 0) return 1;
  close(diagnostics);
  try {
    if (argc < 4) throw std::runtime_error("usage: worker-mux PROGRAM MODEL_DIR CPU_ID...");
    const std::string program = std::filesystem::canonical(argv[1]);
    const std::string model = std::filesystem::canonical(argv[2]);
    std::vector<unsigned> cpus;
    for (int argument = 3; argument < argc; ++argument) {
      const char *text = argv[argument];
      if (*text == '\0' || strspn(text, "0123456789") != strlen(text))
        throw std::runtime_error("CPU ID must be a nonnegative decimal integer");
      char *end;
      errno = 0;
      unsigned long cpu = strtoul(text, &end, 10);
      if (errno == ERANGE || *end || cpu >= CPU_SETSIZE)
        throw std::runtime_error("CPU ID exceeds affinity mask range");
      if (std::find(cpus.begin(), cpus.end(), cpu) != cpus.end())
        throw std::runtime_error("duplicate CPU ID");
      cpus.push_back(static_cast<unsigned>(cpu));
    }
    const unsigned count = cpus.size();
    terminal = std::make_unique<Terminal>();
    nonblocking(STDIN_FILENO); nonblocking(STDOUT_FILENO);
    std::array<char, sizeof(magic) - 1> received{};
    transfer(STDIN_FILENO, received.data(), received.size(), false);
    if (memcmp(received.data(), magic, received.size()))
      throw std::runtime_error("invalid BBMUX1 handshake");
    char directory[] = "/tmp/bbmux-XXXXXX";
    if (!mkdtemp(directory)) throw error("mkdtemp");
    for (unsigned rank = 0; rank < count; ++rank) {
      workers.emplace_back(std::make_unique<Worker>());
      auto path = std::filesystem::path(directory) / ("rank" + std::to_string(rank));
      std::filesystem::create_directory(path);
      spawn(*workers.back(), program, model, rank, cpus[rank], path);
    }
    for (unsigned rank = 0; rank < count; ++rank)
      workers[rank]->thread = std::thread(run, std::ref(*workers[rank]), rank);
    transfer(STDOUT_FILENO, const_cast<char *>(magic), sizeof(magic) - 1, true);
    while (!stopping.load()) {
      std::array<unsigned char, 32> header{};
      try { transfer(STDIN_FILENO, header.data(), header.size(), false); }
      catch (const std::exception &e) {
        if (std::string(e.what()) == "EOF") { status = 0; break; }
        throw;
      }
      uint64_t rank = decode(header.data()), sequence = decode(header.data() + 8);
      uint64_t inputBytes = decode(header.data() + 16), outputBytes = decode(header.data() + 24);
      if (rank == UINT64_MAX && sequence == 0 && inputBytes == 0 && outputBytes == 0) {
        // Synchronize with the final response write and its delivered callback.
        { std::lock_guard<std::mutex> delivered(outputMutex); }
        for (auto &worker : workers) {
          std::unique_lock<std::mutex> lock(worker->mutex);
          if (worker->busy || worker->responding) {
            lock.unlock();
            failure(rank, sequence, "close requires all responses to be delivered");
            throw std::runtime_error("close while workers are active");
          }
        }
        cleanup(workers);
        cleaned = true;
        // The close acknowledgement certifies that every child has been reaped.
        stopping.store(false);
        std::vector<unsigned char> empty;
        response(rank, sequence, 0, empty);
        status = 0;
        break;
      }
      if (rank >= count || inputBytes == 0 || inputBytes > maxBytes ||
          outputBytes == 0 || outputBytes > maxBytes) {
        failure(rank, sequence, "invalid rank or frame length");
        throw std::runtime_error("invalid request frame");
      }
      std::vector<unsigned char> payload(inputBytes);
      transfer(STDIN_FILENO, payload.data(), payload.size(), false);
      auto &w = *workers[rank];
      std::unique_lock<std::mutex> lock(w.mutex);
      if (w.busy) {
        lock.unlock();
        failure(rank, sequence, "rank already has an outstanding request");
        throw std::runtime_error("duplicate rank request");
      }
      w.busy = true; w.available = true; w.sequence = sequence;
      w.resultBytes = outputBytes; w.request = std::move(payload);
      w.wake.notify_one();
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "worker-mux: %s\n", e.what());
  }
  if (!cleaned) cleanup(workers);
  terminal.reset();
  return status;
}

#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

struct Request {
  std::vector<int64_t> tokens, eos;
  size_t maxTokens;
  double temperature;
};

class Execution {
  struct State;
  std::unique_ptr<State> state;

public:
  Execution(const std::filesystem::path &directory,
            const std::filesystem::path &inputResource, const char *index);
  ~Execution() noexcept(false);
  const Request &request() const;
  void prefill(const std::vector<int64_t> &tokens);
  void decode(int64_t token);
  std::span<const float> logits() const;
  void close();
};

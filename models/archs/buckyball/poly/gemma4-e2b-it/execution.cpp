//===- execution.cpp ----------------------------------------------===//
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

#include "execution_state.h"
#include "gemma-parameters.h"
#include "gemma-plan.h"
#include <algorithm>
#include <stdexcept>
#include <sys/time.h>

extern "C" double _mlir_ciface_rtclock() {
  timeval time;
  if (gettimeofday(&time, nullptr))
    throw std::runtime_error("gettimeofday failed");
  return time.tv_sec + time.tv_usec * 1.0e-6;
}

Execution::~Execution() noexcept(false) {
  if (!state->closed)
    close();
}

void Execution::prefill(const std::vector<int64_t> &tokens) {
  if (state->closed || state->prefilled || tokens.empty() ||
      tokens.size() > PrefillLength)
    throw std::runtime_error("invalid Gemma prefill");
  int64_t input[PrefillLength] = {}, positions[PrefillLength];
  std::copy(tokens.begin(), tokens.end(), input);
  for (size_t index = 0; index < PrefillLength; ++index)
    positions[index] = index;
  int64_t valid = tokens.size();
  state->execution->set("prefill.tokens", input, sizeof(input));
  state->execution->set("prefill.positions", positions, sizeof(positions));
  state->execution->set("valid_tokens", &valid, sizeof(valid));
  runPrefill(*state->execution);
  auto logits = state->execution->read("prefill.logits");
  state->output = {static_cast<float *>(logits.data) + logits.offset,
                   MaxVocabSize};
  state->position = tokens.size();
  state->prefilled = true;
}

void Execution::decode(int64_t token) {
  if (state->closed || !state->prefilled || state->position >= CacheLength)
    throw std::runtime_error("invalid Gemma decode");
  state->execution->reset();
  int64_t current = state->position, valid = 1;
  state->execution->set("decode.tokens", &token, sizeof(token));
  state->execution->set("decode.positions", &current, sizeof(current));
  state->execution->set("valid_tokens", &valid, sizeof(valid));
  runDecode(*state->execution);
  auto logits = state->execution->read("decode.logits");
  state->output = {static_cast<float *>(logits.data) + logits.offset,
                   MaxVocabSize};
  ++state->position;
}

void Execution::close() {
  if (state->closed)
    throw std::runtime_error("Gemma execution already closed");
  state->execution.reset();
  state->output = {};
  state->closed = true;
}

const Request &Execution::request() const { return state->request; }
std::span<const float> Execution::logits() const { return state->output; }

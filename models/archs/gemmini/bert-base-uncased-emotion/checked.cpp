#include "model.h"
#include "resources.h"
#include "testutils.h"
#include <algorithm>
#include <cstdlib>
#include <vector>

void check_mapped_bert(const std::vector<std::string> &paths,
                       const int64_t *input, const float *gold) {
  const auto &files = resources::image.files;
  resources::Mapping fp(paths.at(0), files.at(paths.at(0)).bytes, 64);
  resources::Mapping bi(paths.at(1), files.at(paths.at(1)).bytes, 64);
  resources::Mapping wt(paths.at(2), files.at(paths.at(2)).bytes, 64);
  const auto *f = static_cast<const float *>(fp.data());
  const auto *b = static_cast<const int32_t *>(bi.data());
  const auto *w = static_cast<const int8_t *>(wt.data());
  bert::Parameters parameters{
      f, f, 0, {int64_t(files.at(paths[0]).bytes / sizeof(*f))}, {1}};
  bert::Bias biases{
      b, b, 0, {int64_t(files.at(paths[1]).bytes / sizeof(*b))}, {1}};
  bert::Weights weights{w, w, 0, {int64_t(files.at(paths[2]).bytes)}, {1}};
  bert::Tokens ids{input, input, 0, {1, 32}, {32, 1}};
  bert::Tokens positions{input + 32, input + 32, 0, {1, 32}, {32, 1}};
  bert::Tokens types{input + 64, input + 64, 0, {1, 32}, {32, 1}};
  bert::Tokens mask{input + 96, input + 96, 0, {1, 32}, {32, 1}};
  bert::Output output{};
  const auto start = read_counter();
  _mlir_ciface_forward(&output, &parameters, &biases, &weights, &ids,
                       &positions, &types, &mask);
  printf("Inference cycles: %lu\n", (unsigned long)(read_counter() - start));
  const auto *values = output.data();
  const bool different = check_output(values, gold, 6, 2e-2f) ||
                         std::max_element(values, values + 6) - values !=
                             std::max_element(gold, gold + 6) - gold;
  std::free(output.allocated);
  if (different)
    throw std::runtime_error("BERT logits differ from the reference");
}

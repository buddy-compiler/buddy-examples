#include "BuddyTraceUtils.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace {
constexpr int64_t kShapeInput[] = {1, 1, 28, 28};
constexpr int64_t kShapeConv1[] = {1, 6, 24, 24};
constexpr int64_t kShapePool1[] = {1, 6, 12, 12};
constexpr int64_t kShapeConv2[] = {1, 16, 8, 8};
constexpr int64_t kShapePool2[] = {1, 16, 4, 4};
constexpr int64_t kShapeFlat[] = {1, 256};
constexpr int64_t kShapeFc1[] = {1, 120};
constexpr int64_t kShapeFc2[] = {1, 84};
constexpr int64_t kShapeFc3[] = {1, 10};

constexpr BuddyTraceMeta kMeta[] = {
    {"input_nchw", "nchw", kShapeInput, 4},
    {"conv1_out_nchw", "nchw", kShapeConv1, 4},
    {"relu1_out_nchw", "nchw", kShapeConv1, 4},
    {"pool1_out_nchw", "nchw", kShapePool1, 4},
    {"conv2_out_nchw", "nchw", kShapeConv2, 4},
    {"relu2_out_nchw", "nchw", kShapeConv2, 4},
    {"pool2_out_nchw", "nchw", kShapePool2, 4},
    {"flatten_out", "nc", kShapeFlat, 2},
    {"fc1_out", "nc", kShapeFc1, 2},
    {"relu3_out", "nc", kShapeFc1, 2},
    {"fc2_out", "nc", kShapeFc2, 2},
    {"relu4_out", "nc", kShapeFc2, 2},
    {"fc3_out", "nc", kShapeFc3, 2},
};
} // namespace

extern "C" const BuddyTraceMeta *buddyGetTraceMeta(size_t *count) {
  if (!count) {
    fprintf(stderr, "LeNet trace config error: count is null\n");
    std::abort();
  }
  *count = std::size(kMeta);
  return kMeta;
}

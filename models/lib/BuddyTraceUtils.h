#ifndef MODELTEST_BUDDY_TRACE_UTILS_H
#define MODELTEST_BUDDY_TRACE_UTILS_H

#include <cstddef>
#include <cstdint>

struct BuddyTraceMeta {
  const char *tag;
  const char *layout;
  const int64_t *shape;
  int64_t rank;
};

extern "C" const BuddyTraceMeta *buddyGetTraceMeta(size_t *count);

#endif

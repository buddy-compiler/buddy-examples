#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace model {
inline const char *imageFile(uintptr_t base, const char *name,
                             uint64_t *bytes) {
  const char *line = strchr(reinterpret_cast<const char *>(base), '\n') + 1;
  const size_t length = strlen(name);
  while (*line != '\n') {
    const char *end = strchr(line, '\n');
    if (!strncmp(line, name, length) && line[length] == '\t') {
      char *next;
      const uint64_t offset = strtoull(line + length + 1, &next, 10);
      *bytes = strtoull(next + 1, nullptr, 10);
      return reinterpret_cast<const char *>(base) + offset;
    }
    line = end + 1;
  }
  printf("Model file missing: %s\n", name);
  exit(1);
}
} // namespace model

#pragma once

#include <dlfcn.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace model {
using SharedLibrary = std::unique_ptr<void, int (*)(void *)>;

inline SharedLibrary openLibrary(const std::string &path) {
  void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!handle)
    throw std::runtime_error("Cannot load " + path + ": " + dlerror());
  return SharedLibrary(handle, dlclose);
}
} // namespace model

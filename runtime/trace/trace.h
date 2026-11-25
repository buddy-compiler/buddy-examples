// Shared by the tensor traces and the two backends of the cycle traces.
#pragma once

#include "location.h"
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <utility>

constexpr int64_t kTraceMaxId = 4096;
constexpr int64_t kTraceMaxPathDepth = 4;

// Serializes the trace files a process writes.
extern std::mutex traceMutex;

[[noreturn]] void traceError(const char *message);
uint64_t readCycle();
const char *counterName();
const char *platformName();
std::pair<uint64_t, uint64_t> processThread();
std::filesystem::path traceRoot(thread_location_t location);
void writeTracePath(char *buffer, size_t size, int64_t depth, int64_t path0,
                    int64_t path1, int64_t path2, int64_t path3);
FILE *openTraceFilePath(const char *kind, int64_t depth, int64_t path0,
                        int64_t path1, int64_t path2, int64_t path3,
                        const char *mode);

// The cycle traces of the backend the program links.
void beginCycle(int64_t id, int64_t depth, int64_t p0, int64_t p1, int64_t p2,
                int64_t p3);
void endCycle(int64_t id, int64_t depth, int64_t p0, int64_t p1, int64_t p2,
              int64_t p3);

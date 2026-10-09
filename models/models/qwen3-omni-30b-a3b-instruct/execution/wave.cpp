#include "wave.h"
#include "embedding.h"
#include "ffn/params.h"
#include <cstdlib>
#include <fstream>
#include <runtime.h>

using namespace WaveParams;
static constexpr size_t workspaceBytes = size_t(2) * 1024 * 1024 * 1024;

namespace {
struct SpanJob {
  Hidden input, output;
  Floats *floats;
  Bytes *bytes;
  Convolution kernel;
  task *pending;
  size_t start;
  SpanJob(const Step &step, size_t first, Floats *fp, Bytes *packed)
      : input({1, step.inputChannels, step.ownedRows + step.halo}, 0.0f),
        output({1, step.outputChannels, step.ownedRows}, false, 0), floats(fp),
        bytes(packed), kernel(step.run), start(first) {}
};
void run_span(void *argument) {
  auto &job = *static_cast<SpanJob *>(argument);
  job.kernel(&job.output, job.floats, job.bytes, &job.input);
}
} // namespace

Wave::Wave(const std::filesystem::path &directory) {
  std::ifstream input;
  input.exceptions(std::ios::failbit | std::ios::badbit);
  input.open(directory / "wave-layout.bin", std::ios::binary);
  std::array<uint64_t, 4> header;
  input.read(reinterpret_cast<char *>(header.data()), sizeof(header));
  if (header[0] != 0x574156450001 || header[3] != 2 + 2 * layers + stepCount)
    throw std::runtime_error(
        "Code2Wav weight layout does not match compiled model");
  floats.reset(new float[header[1]]);
  bytes.reset(new int8_t[header[2]]);
  regions.resize(header[3]);
  input.read(reinterpret_cast<char *>(regions.data()),
             regions.size() * sizeof(Region));
  input.close();
  input.open(directory / "wave.f32", std::ios::binary);
  input.read(reinterpret_cast<char *>(floats.get()), header[1] * sizeof(float));
  input.close();
  input.open(directory / "wave.bin", std::ios::binary);
  input.read(reinterpret_cast<char *>(bytes.get()), header[2]);
  runtime_init(1024 * 1024);
  workspace = aligned_alloc(64, workspaceBytes);
  if (!workspace)
    throw std::bad_alloc();
  workspace_init(workspace, workspaceBytes);
  std::cerr << "Code2Wav ready: layers=" << layers
            << " weight_bytes=" << header[1] * 4 + header[2] << '\n';
}
Wave::~Wave() { free(workspace); }

std::pair<View<float, 1>, View<int8_t, 1>> Wave::parameters(size_t index) {
  const auto &region = regions.at(index);
  return {
      View<float, 1>(floats.get() + region.float_offset, {region.float_count}),
      View<int8_t, 1>(bytes.get() + region.byte_offset, {region.byte_count})};
}

void Wave::execute(const Command &command) {
  auto [operation, count, unused0, unused1] = command;
  if (operation != 22 || !count || count > buckets[7] || unused0 || unused1)
    throw std::runtime_error("invalid Code2Wav command");
  size_t bucket = 0;
  while (buckets[bucket] < count)
    ++bucket;
  size_t length = buckets[bucket];
  std::vector<uint64_t> codes(groups * count);
  read_values(codes.data(), codes.size());
  std::vector<float> hidden(length * width, 0.0f);
  const auto &embedding = regions[0];
  if (embedding.float_count ||
      embedding.byte_count != groups * vocabulary * (width + width / 32))
    throw std::runtime_error("Code2Wav embedding requires MXFP8 row storage");
  std::vector<uint64_t> selected(count * groups);
  for (size_t token = 0; token < count; ++token)
    for (size_t group = 0; group < groups; ++group) {
      const auto code = codes[group * count + token];
      if (code >= vocabulary)
        throw std::runtime_error("invalid acoustic code ID");
      selected[token * groups + group] = group * vocabulary + code;
    }
  std::vector<float> decoded(count * groups * width);
  embedding_rows(decoded.data(), bytes.get() + embedding.byte_offset, width,
                 selected.data(), selected.size());
  for (size_t token = 0; token < count; ++token)
    for (size_t group = 0; group < groups; ++group)
      for (size_t column = 0; column < width; ++column)
        hidden[token * width + column] +=
            decoded[(token * groups + group) * width + column];
  for (size_t token = 0; token < count; ++token)
    for (size_t column = 0; column < width; ++column)
      hidden[token * width + column] *= 1.0f / groups;
  Slots positions({length});
  for (size_t index = 0; index < length; ++index)
    positions[index] = index;
  for (size_t layer = 0; layer < layers; ++layer) {
    workspace_begin(workspace, workspaceBytes);
    View<float, 2> input(hidden.data(), {length, width});
    Matrix result({length, width}, false, 0);
    auto [fp, packed] = parameters(1 + 2 * layer);
    attention[bucket](&result, &fp, &packed, &input, &positions);
    std::copy_n(result.getData(), hidden.size(), hidden.data());
    workspace_free(result.release());
    workspace_begin(workspace, workspaceBytes);
    auto [df, dp] = parameters(2 + 2 * layer);
    dense[bucket](&result, &df, &dp, &input);
    std::copy_n(result.getData(), hidden.size(), hidden.data());
    workspace_free(result.release());
  }
  workspace_begin(workspace, workspaceBytes);
  View<float, 2> input(hidden.data(), {length, width});
  Matrix normalized({length, width}, false, 0);
  auto [nf, np] = parameters(1 + 2 * layers);
  norm[bucket](&normalized, &nf, &input);
  std::vector<float> samples(hidden.size());
  for (size_t channel = 0; channel < width; ++channel)
    for (size_t token = 0; token < length; ++token)
      samples[channel * length + token] = normalized[token * width + channel];
  workspace_free(normalized.release());
  for (size_t index = 0; index < stepCount; ++index) {
    workspace_begin(workspace, workspaceBytes);
    const auto &step = steps[bucket][index];
    auto [fp, packed] = parameters(2 + 2 * layers + index);
    if (step.ownedRows) {
      std::vector<std::unique_ptr<SpanJob>> jobs;
      for (size_t start = 0; start < step.inputLength;
           start += step.ownedRows) {
        auto job = std::make_unique<SpanJob>(step, start, &fp, &packed);
        size_t first = start > step.halo ? start - step.halo : 0;
        size_t last = std::min(start + step.ownedRows, step.inputLength);
        size_t offset = step.halo + first - start;
        for (size_t channel = 0; channel < step.inputChannels; ++channel)
          std::copy_n(samples.data() + channel * step.inputLength + first,
                      last - first,
                      job->input.getData() +
                          channel * (step.ownedRows + step.halo) + offset);
        job->pending = task_submit(CORE_SIGNATURE, run_span, job.get());
        jobs.push_back(std::move(job));
      }
      for (auto &job : jobs) {
        if (task_wait(job->pending))
          throw std::runtime_error("Code2Wav residual span task failed");
        size_t rows = std::min(step.ownedRows, step.outputLength - job->start);
        for (size_t channel = 0; channel < step.outputChannels; ++channel)
          std::copy_n(job->output.getData() + channel * step.ownedRows, rows,
                      samples.data() + channel * step.outputLength +
                          job->start);
        workspace_free(job->output.release());
      }
      continue;
    }
    View<float, 3> data(samples.data(),
                        {1, step.inputChannels, step.inputLength});
    Hidden result({1, step.outputChannels, step.outputLength}, false, 0);
    step.run(&result, &fp, &packed, &data);
    samples.assign(result.getData(),
                   result.getData() + step.outputChannels * step.outputLength);
    workspace_free(result.release());
  }
  for (size_t sample = 0; sample < count * totalUpsample - tail; ++sample)
    samples[sample] = std::clamp(samples[sample], -1.0f, 1.0f);
  write_values(samples.data(), count * totalUpsample - tail);
}

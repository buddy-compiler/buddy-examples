#include "SharedLibrary.h"
#include <buddy/runtime/core/InferenceRunner.h>
#include <buddy/runtime/core/ModelManifest.h>
#include <vector>
//===- buddy-minimax-h3-ref2va-runner.cpp
//------------------------------------===//
//
// MiniMax-H3 Ref2VA host runtime: Context-IR API -> Buddy Base 768p ->
// Regenerate-2K API.
//
//===----------------------------------------------------------------------===//

#include "model_parameters.h"
#include <buddy/Core/Container.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace buddy;

namespace {

using namespace model_parameters;

struct VideoAudio {
  MemRef<float, 5> video;
  MemRef<float, 3> audio;
  VideoAudio(MemRef<float, 5> v, MemRef<float, 3> a) : video(v), audio(a) {}
};

using Fn_mlir_ciface_forward_text_encoder =
    void (*)(MemRef<float, 3> *result, MemRef<float, 1> *params,
             MemRef<long long, 2> *ids, MemRef<long long, 2> *mask);

using Fn_mlir_ciface_forward_transformer = void (*)(VideoAudio *result,
                                                    MemRef<float, 1> *params,
                                                    MemRef<float, 5> *video,
                                                    MemRef<float, 1> *timestep,
                                                    MemRef<float, 3> *text,
                                                    MemRef<float, 3> *audio);

using Fn_mlir_ciface_forward_visual_vae = void (*)(MemRef<float, 5> *result,
                                                   MemRef<float, 1> *params,
                                                   MemRef<float, 5> *latents);

using Fn_mlir_ciface_forward_visual_vae_encode =
    void (*)(MemRef<float, 5> *result, MemRef<float, 1> *params,
             MemRef<float, 5> *frames);

using Fn_mlir_ciface_forward_audio_vae = void (*)(MemRef<float, 3> *result,
                                                  MemRef<float, 1> *params,
                                                  MemRef<float, 3> *latents);

void runOrDie(const std::string &cmd) {
  std::cout << "[cmd] " << cmd << std::endl;
  int rc = std::system(cmd.c_str());
  if (rc != 0)
    throw std::runtime_error("command failed (" + std::to_string(rc) +
                             "): " + cmd);
}

size_t floatCount(const std::string &path) {
  auto bytes = std::filesystem::file_size(path);
  if (bytes % sizeof(float) != 0)
    throw std::runtime_error("param file size not aligned: " + path);
  size_t n = bytes / sizeof(float);
  if (n == 0)
    throw std::runtime_error("empty param file: " + path);
  return n;
}

void loadFloatParams(const std::string &path, MemRef<float, 1> &dst) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("failed to open: " + path);
  f.read(reinterpret_cast<char *>(dst.getData()),
         dst.getSize() * sizeof(float));
  if (!f)
    throw std::runtime_error("failed to read: " + path);
}

void fillNormal(float *data, size_t n, unsigned seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (size_t i = 0; i < n; ++i)
    data[i] = dist(gen);
}

// Flow-matching schedule with shift (sigma -> t = 1 - sigma).
std::vector<float> makeTimesteps(int steps, float shift) {
  if (steps < 1)
    throw std::runtime_error("denoise steps must be positive");
  std::vector<float> ts(steps);
  for (int i = 0; i < steps; ++i) {
    float u = static_cast<float>(i) / static_cast<float>(steps);
    float sigma = shift * u / (1.0f + (shift - 1.0f) * u);
    ts[i] = 1.0f - sigma;
  }
  return ts;
}

std::string shellQuote(const std::string &value) {
  std::string quoted = "'";
  for (char c : value)
    quoted += c == '\'' ? "'\"'\"'" : std::string(1, c);
  return quoted + "'";
}

void writeRawAndMux(MemRef<float, 5> &video, MemRef<float, 3> &audio,
                    const std::string &mp4Path) {
  const std::string rgbPath = mp4Path + ".rgb";
  const std::string pcmPath = mp4Path + ".pcm";
  {
    std::ofstream rgb(rgbPath, std::ios::binary);
    if (!rgb)
      throw std::runtime_error("failed to open " + rgbPath);
    size_t n = static_cast<size_t>(1) * 3 * FRAMES * HEIGHT * WIDTH;
    std::vector<uint8_t> bytes(n);
    for (size_t i = 0; i < n; ++i) {
      float x = video.getData()[i];
      int v = static_cast<int>(x * 255.0f);
      if (v < 0)
        v = 0;
      if (v > 255)
        v = 255;
      bytes[i] = static_cast<uint8_t>(v);
    }
    rgb.write(reinterpret_cast<char *>(bytes.data()), bytes.size());
  }
  {
    std::ofstream pcm(pcmPath, std::ios::binary);
    if (!pcm)
      throw std::runtime_error("failed to open " + pcmPath);
    size_t n = static_cast<size_t>(SAMPLE_RATE) * DURATION_S * 2;
    std::vector<int16_t> samples(n);
    // audio memref layout assumed [1, C, T] float in [-1, 1]; expand/truncate.
    size_t srcN = audio.getSize();
    for (size_t i = 0; i < n; ++i) {
      float x = audio.getData()[i % srcN];
      int v = static_cast<int>(x * 32767.0f);
      if (v < -32768)
        v = -32768;
      if (v > 32767)
        v = 32767;
      samples[i] = static_cast<int16_t>(v);
    }
    pcm.write(reinterpret_cast<char *>(samples.data()),
              samples.size() * sizeof(int16_t));
  }
  std::ostringstream cmd;
  cmd << "ffmpeg -y -f rawvideo -pix_fmt rgb24 -s " << WIDTH << "x" << HEIGHT
      << " -r " << FPS << " -i " << shellQuote(rgbPath) << " -f s16le -ar "
      << SAMPLE_RATE << " -ac 2 -i " << shellQuote(pcmPath)
      << " -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "
      << shellQuote(mp4Path) << " >/dev/null 2>&1";
  runOrDie(cmd.str());
}

} // namespace

namespace {
class MiniMaxRefRunner final : public buddy::runtime::InferenceRunner {
public:
  void run(const buddy::runtime::RunConfig &runConfig) override {
    const auto manifest =
        buddy::runtime::ModelManifest::loadFromRax(runConfig.raxPath);
    std::vector<model::SharedLibrary> runtimes;
    for (const auto &path : manifest.dependentSoPaths)
      runtimes.push_back(model::openLibrary(path));
    auto library = model::openLibrary(manifest.soPath);
    auto _mlir_ciface_forward_text_encoder =
        reinterpret_cast<Fn_mlir_ciface_forward_text_encoder>(
            dlsym(library.get(), "_mlir_ciface_forward_text_encoder"));
    if (!_mlir_ciface_forward_text_encoder)
      throw std::runtime_error("Missing _mlir_ciface_forward_text_encoder: " +
                               std::string(dlerror()));
    auto _mlir_ciface_forward_transformer =
        reinterpret_cast<Fn_mlir_ciface_forward_transformer>(
            dlsym(library.get(), "_mlir_ciface_forward_transformer"));
    if (!_mlir_ciface_forward_transformer)
      throw std::runtime_error("Missing _mlir_ciface_forward_transformer: " +
                               std::string(dlerror()));
    auto _mlir_ciface_forward_visual_vae =
        reinterpret_cast<Fn_mlir_ciface_forward_visual_vae>(
            dlsym(library.get(), "_mlir_ciface_forward_visual_vae"));
    if (!_mlir_ciface_forward_visual_vae)
      throw std::runtime_error("Missing _mlir_ciface_forward_visual_vae: " +
                               std::string(dlerror()));
    auto _mlir_ciface_forward_visual_vae_encode =
        reinterpret_cast<Fn_mlir_ciface_forward_visual_vae_encode>(
            dlsym(library.get(), "_mlir_ciface_forward_visual_vae_encode"));
    if (!_mlir_ciface_forward_visual_vae_encode)
      throw std::runtime_error(
          "Missing _mlir_ciface_forward_visual_vae_encode: " +
          std::string(dlerror()));
    auto _mlir_ciface_forward_audio_vae =
        reinterpret_cast<Fn_mlir_ciface_forward_audio_vae>(
            dlsym(library.get(), "_mlir_ciface_forward_audio_vae"));
    if (!_mlir_ciface_forward_audio_vae)
      throw std::runtime_error("Missing _mlir_ciface_forward_audio_vae: " +
                               std::string(dlerror()));
    if (runConfig.prompt.empty() || runConfig.outputPath.empty())
      throw std::runtime_error("--prompt and --output are required");
    const auto outputDir = std::filesystem::absolute(runConfig.outputPath);
    std::filesystem::create_directories(outputDir);
    const std::string prompt = runConfig.prompt;
    const std::string apiPy = manifest.weightPaths.at(4);
    if (runConfig.imagePath.empty())
      throw std::runtime_error("--image is required");
    const std::vector<std::string> refImages{runConfig.imagePath};
    // 1) Context-IR
    std::string contextPath = (outputDir / "h3_context_prompt.txt").string();
    {
      std::ostringstream cmd;
      cmd << "python3 " << shellQuote(apiPy) << " context-ir --prompt "
          << shellQuote(prompt) << " --duration " << DURATION_S
          << " --ratio 16:9 --out " << shellQuote(contextPath);
      for (const auto &img : refImages) {
        if (!std::filesystem::exists(img))
          throw std::runtime_error("ref image not found: " + img);
        cmd << " --image " << shellQuote(img) << " --image-role reference";
      }
      runOrDie(cmd.str());
    }
    {
      std::ifstream in(contextPath);
      if (!in)
        throw std::runtime_error("failed to read " + contextPath);
      std::string expanded(std::istreambuf_iterator<char>(in), {});
      if (expanded.empty())
        throw std::runtime_error("empty Context-IR prompt");
    }

    std::string pText = manifest.weightPaths.at(0);
    std::string pTr = manifest.weightPaths.at(1);
    std::string pV = manifest.weightPaths.at(2);
    std::string pA = manifest.weightPaths.at(3);
    const std::string pVE = manifest.weightPaths.at(5);
    for (auto &p : {pText, pTr, pV, pA}) {
      if (!std::filesystem::exists(p))
        throw std::runtime_error("missing param file: " + p);
    }
    MemRef<float, 1> arg0_text({floatCount(pText)});
    MemRef<float, 1> arg0_tr({floatCount(pTr)});
    MemRef<float, 1> arg0_v({floatCount(pV)});
    MemRef<float, 1> arg0_a({floatCount(pA)});
    MemRef<float, 1> arg0_ve({floatCount(pVE)});
    loadFloatParams(pText, arg0_text);
    loadFloatParams(pTr, arg0_tr);
    loadFloatParams(pV, arg0_v);
    loadFloatParams(pA, arg0_a);
    loadFloatParams(pVE, arg0_ve);

    std::string idsPath = (outputDir / "input_ids.bin").string();
    std::string maskPath = (outputDir / "attention_mask.bin").string();
    {
      const std::string model = "MiniMaxAI/MiniMax-H3";
      std::ostringstream cmd;
      cmd << "python3 " << shellQuote(apiPy) << " tokenize --model "
          << shellQuote(model) << " --subfolder Ref2VA --prompt-file "
          << shellQuote(contextPath) << " --ids-out " << shellQuote(idsPath)
          << " --mask-out " << shellQuote(maskPath) << " --max-len "
          << TEXT_LEN;
      runOrDie(cmd.str());
    }
    MemRef<long long, 2> inputIds({1, TEXT_LEN});
    MemRef<long long, 2> attnMask({1, TEXT_LEN});
    {
      std::ifstream in(idsPath, std::ios::binary);
      if (!in)
        throw std::runtime_error("failed to open " + idsPath);
      in.read(reinterpret_cast<char *>(inputIds.getData()),
              TEXT_LEN * sizeof(long long));
      if (!in)
        throw std::runtime_error("failed to read " + idsPath);
    }
    {
      std::ifstream in(maskPath, std::ios::binary);
      if (!in)
        throw std::runtime_error("failed to open " + maskPath);
      in.read(reinterpret_cast<char *>(attnMask.getData()),
              TEXT_LEN * sizeof(long long));
      if (!in)
        throw std::runtime_error("failed to read " + maskPath);
    }

    MemRef<float, 3> textEmb({1, TEXT_LEN, TEXT_DIM});
    auto t0 = std::chrono::high_resolution_clock::now();
    _mlir_ciface_forward_text_encoder(&textEmb, &arg0_text, &inputIds,
                                      &attnMask);

    MemRef<float, 5> videoLatent({1, LATENT_C, LATENT_T, LATENT_H, LATENT_W});
    MemRef<float, 3> audioLatent({1, AUDIO_LATENT_C, AUDIO_LATENT_T});
    fillNormal(videoLatent.getData(), videoLatent.getSize(), 0);
    fillNormal(audioLatent.getData(), audioLatent.getSize(), 1);

    {
      std::string frameBin = (outputDir / "cond_frames.bin").string();
      if (!std::filesystem::exists(frameBin))
        throw std::runtime_error(
            "ref2va requires cond_frames.bin (1x3xTxHxW float32)");
      MemRef<float, 5> frames({1, 3, FRAMES, HEIGHT, WIDTH});
      std::ifstream in(frameBin, std::ios::binary);
      in.read(reinterpret_cast<char *>(frames.getData()),
              frames.getSize() * sizeof(float));
      if (!in)
        throw std::runtime_error("failed to read cond_frames.bin");
      MemRef<float, 5> condLatent({1, LATENT_C, LATENT_T, LATENT_H, LATENT_W});
      _mlir_ciface_forward_visual_vae_encode(&condLatent, &arg0_ve, &frames);
      std::memcpy(videoLatent.getData(), condLatent.getData(),
                  videoLatent.getSize() * sizeof(float));
    }

    auto timesteps = makeTimesteps(runConfig.inferenceSteps, VIDEO_SHIFT);
    MemRef<float, 5> videoOut({1, LATENT_C, LATENT_T, LATENT_H, LATENT_W});
    MemRef<float, 3> audioOut({1, AUDIO_LATENT_C, AUDIO_LATENT_T});
    VideoAudio result(videoOut, audioOut);
    MemRef<float, 1> tMem({1});
    for (int step = 0; step < DENOISE_STEPS; ++step) {
      tMem.getData()[0] = timesteps[step];
      _mlir_ciface_forward_transformer(&result, &arg0_tr, &videoLatent, &tMem,
                                       &textEmb, &audioLatent);
      // Euler: x <- x + (x1 - x) * dt with velocity = clean - noise prediction.
      float dt = (step + 1 < DENOISE_STEPS)
                     ? (timesteps[step + 1] - timesteps[step])
                     : (0.0f - timesteps[step]);
      for (size_t i = 0; i < videoLatent.getSize(); ++i)
        videoLatent.getData()[i] += result.video.getData()[i] * dt;
      for (size_t i = 0; i < audioLatent.getSize(); ++i)
        audioLatent.getData()[i] += result.audio.getData()[i] * dt;
      std::cout << "[denoise] step " << step << "/" << DENOISE_STEPS
                << std::endl;
    }

    MemRef<float, 5> framesOut({1, 3, FRAMES, HEIGHT, WIDTH});
    MemRef<float, 3> wavOut({1, 2, SAMPLE_RATE * DURATION_S});
    _mlir_ciface_forward_visual_vae(&framesOut, &arg0_v, &videoLatent);
    _mlir_ciface_forward_audio_vae(&wavOut, &arg0_a, &audioLatent);

    const std::string mp4_768 = (outputDir / "h3_ref2va_768p.mp4").string();
    writeRawAndMux(framesOut, wavOut, mp4_768);

    // 3) Regenerate-2K
    const std::string mp4_2k = (outputDir / "h3_ref2va_2k.mp4").string();
    {
      std::ostringstream cmd;
      cmd << "python3 " << shellQuote(apiPy) << " regenerate-2k --prompt-file "
          << shellQuote(contextPath) << " --video " << shellQuote(mp4_768)
          << " --out " << shellQuote(mp4_2k);
      runOrDie(cmd.str());
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[Output] 768p: " << shellQuote(mp4_768) << std::endl;
    std::cout << "[Output] 2K:   " << shellQuote(mp4_2k) << std::endl;
    std::cout << "[Log] elapsed_s=" << sec << std::endl;
  }
};

} // namespace
extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new MiniMaxRefRunner();
}
extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

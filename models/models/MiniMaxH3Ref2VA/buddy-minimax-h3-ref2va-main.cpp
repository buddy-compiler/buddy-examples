//===- buddy-minimax-h3-ref2va-main.cpp ------------------------------------===//
//
// MiniMax-H3 Ref2VA host runtime: Context-IR API -> Buddy Base 768p ->
// Regenerate-2K API.
//
//===----------------------------------------------------------------------===//

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

constexpr int kDurationS = 4;
constexpr int kFps = 24;
constexpr int kFrames = kDurationS * kFps;
constexpr int kHeight = 768;
constexpr int kWidth = 1344;
constexpr int kLatentT = kFrames / 4;
constexpr int kLatentH = kHeight / 16;
constexpr int kLatentW = kWidth / 16;
constexpr int kLatentC = 24;
constexpr int kAudioLatentC = 32;
constexpr int kAudioLatentT = kDurationS * 40;
constexpr int kTextLen = 512;
constexpr int kTextDim = 5120;
constexpr int kSampleRate = 32000;
constexpr int kDenoiseSteps = 30;
constexpr float kVideoShift = 12.0f;

struct VideoAudio {
  MemRef<float, 5> video;
  MemRef<float, 3> audio;
  VideoAudio(MemRef<float, 5> v, MemRef<float, 3> a) : video(v), audio(a) {}
};

extern "C" void _mlir_ciface_forward_text_encoder(MemRef<float, 3> *result,
                                                  MemRef<float, 1> *params,
                                                  MemRef<long long, 2> *ids,
                                                  MemRef<long long, 2> *mask);

extern "C" void _mlir_ciface_forward_transformer(VideoAudio *result,
                                                 MemRef<float, 1> *params,
                                                 MemRef<float, 5> *video,
                                                 MemRef<float, 1> *timestep,
                                                 MemRef<float, 3> *text,
                                                 MemRef<float, 3> *audio);

extern "C" void _mlir_ciface_forward_visual_vae(MemRef<float, 5> *result,
                                                MemRef<float, 1> *params,
                                                MemRef<float, 5> *latents);

extern "C" void
_mlir_ciface_forward_visual_vae_encode(MemRef<float, 5> *result,
                                       MemRef<float, 1> *params,
                                       MemRef<float, 5> *frames);

extern "C" void _mlir_ciface_forward_audio_vae(MemRef<float, 3> *result,
                                               MemRef<float, 1> *params,
                                               MemRef<float, 3> *latents);

std::string envOrDie(const char *key) {
  const char *v = std::getenv(key);
  if (!v || !*v)
    throw std::runtime_error(std::string("missing env var: ") + key);
  return v;
}

void runOrDie(const std::string &cmd) {
  std::cout << "[cmd] " << cmd << std::endl;
  int rc = std::system(cmd.c_str());
  if (rc != 0)
    throw std::runtime_error("command failed (" + std::to_string(rc) + "): " +
                             cmd);
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

void writeRawAndMux(const MemRef<float, 5> &video, const MemRef<float, 3> &audio,
                    const std::string &mp4Path) {
  const std::string rgbPath = "h3_768p.rgb";
  const std::string pcmPath = "h3_768p.pcm";
  {
    std::ofstream rgb(rgbPath, std::ios::binary);
    if (!rgb)
      throw std::runtime_error("failed to open " + rgbPath);
    size_t n = static_cast<size_t>(1) * 3 * kFrames * kHeight * kWidth;
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
    size_t n = static_cast<size_t>(kSampleRate) * kDurationS * 2;
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
  cmd << "ffmpeg -y -f rawvideo -pix_fmt rgb24 -s " << kWidth << "x" << kHeight
      << " -r " << kFps << " -i " << rgbPath
      << " -f s16le -ar " << kSampleRate << " -ac 2 -i " << pcmPath
      << " -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest " << mp4Path
      << " >/dev/null 2>&1";
  runOrDie(cmd.str());
}

void printUsage(const char *argv0) {
  std::cerr << "Usage: " << argv0
            << " --prompt TEXT --ref-image IMG [--ref-image IMG ...]\n";
}

} // namespace

int main(int argc, char **argv) {
  std::string prompt;
  std::vector<std::string> refImages;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char *flag) -> std::string {
      if (i + 1 >= argc)
        throw std::runtime_error(std::string("missing value for ") + flag);
      return argv[++i];
    };
    if (a == "--prompt")
      prompt = need("--prompt");
    else if (a == "--ref-image")
      refImages.push_back(need("--ref-image"));
    else if (a == "--help" || a == "-h") {
      printUsage(argv[0]);
      return 0;
    } else {
      printUsage(argv[0]);
      throw std::runtime_error("unknown arg: " + a);
    }
  }
  if (prompt.empty())
    throw std::runtime_error("--prompt is required");
  if (refImages.empty())
    throw std::runtime_error("ref2va requires at least one --ref-image");

  std::string exampleDir = "./";
#ifdef MINIMAX_H3_REF2VA_EXAMPLE_PATH
  exampleDir = MINIMAX_H3_REF2VA_EXAMPLE_PATH;
#endif
  std::string apiPy = exampleDir + "/h3_api.py";
  if (!std::filesystem::exists(apiPy))
    apiPy = "h3_api.py";
  if (!std::filesystem::exists(apiPy))
    throw std::runtime_error("h3_api.py not found");

  // 1) Context-IR
  std::string contextPath = "h3_context_prompt.txt";
  {
    std::ostringstream cmd;
    cmd << "python3 " << apiPy << " context-ir --prompt "
        << std::quoted(prompt) << " --duration " << kDurationS
        << " --ratio 16:9 --out " << contextPath;
    for (const auto &img : refImages) {
      if (!std::filesystem::exists(img))
        throw std::runtime_error("ref image not found: " + img);
      cmd << " --image " << std::quoted(img) << " --image-role reference";
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

  std::string pText = exampleDir + "/arg0_text_encoder.data";
  std::string pTr = exampleDir + "/arg0_transformer.data";
  std::string pV = exampleDir + "/arg0_visual_vae.data";
  std::string pA = exampleDir + "/arg0_audio_vae.data";
  for (auto &p : {pText, pTr, pV, pA}) {
    if (!std::filesystem::exists(p))
      throw std::runtime_error("missing param file: " + p);
  }
  MemRef<float, 1> arg0_text({floatCount(pText)});
  MemRef<float, 1> arg0_tr({floatCount(pTr)});
  MemRef<float, 1> arg0_v({floatCount(pV)});
  MemRef<float, 1> arg0_a({floatCount(pA)});
  loadFloatParams(pText, arg0_text);
  loadFloatParams(pTr, arg0_tr);
  loadFloatParams(pV, arg0_v);
  loadFloatParams(pA, arg0_a);

  std::string idsPath = "input_ids.bin";
  std::string maskPath = "attention_mask.bin";
  {
    const char *modelEnv = std::getenv("MINIMAX_H3_REF2VA_MODEL_PATH");
    std::string model =
        (modelEnv && *modelEnv) ? modelEnv : "MiniMaxAI/MiniMax-H3";
    std::ostringstream cmd;
    cmd << "python3 " << apiPy << " tokenize --model " << std::quoted(model)
        << " --subfolder Ref2VA --prompt-file " << contextPath
        << " --ids-out " << idsPath << " --mask-out " << maskPath
        << " --max-len " << kTextLen;
    runOrDie(cmd.str());
  }
  MemRef<long long, 2> inputIds({1, kTextLen});
  MemRef<long long, 2> attnMask({1, kTextLen});
  {
    std::ifstream in(idsPath, std::ios::binary);
    if (!in)
      throw std::runtime_error("failed to open " + idsPath);
    in.read(reinterpret_cast<char *>(inputIds.getData()),
            kTextLen * sizeof(long long));
    if (!in)
      throw std::runtime_error("failed to read " + idsPath);
  }
  {
    std::ifstream in(maskPath, std::ios::binary);
    if (!in)
      throw std::runtime_error("failed to open " + maskPath);
    in.read(reinterpret_cast<char *>(attnMask.getData()),
            kTextLen * sizeof(long long));
    if (!in)
      throw std::runtime_error("failed to read " + maskPath);
  }

  MemRef<float, 3> textEmb({1, kTextLen, kTextDim});
  auto t0 = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward_text_encoder(&textEmb, &arg0_text, &inputIds, &attnMask);

  MemRef<float, 5> videoLatent({1, kLatentC, kLatentT, kLatentH, kLatentW});
  MemRef<float, 3> audioLatent({1, kAudioLatentC, kAudioLatentT});
  fillNormal(videoLatent.getData(), videoLatent.getSize(), 0);
  fillNormal(audioLatent.getData(), audioLatent.getSize(), 1);

  {
    std::string frameBin = exampleDir + "/cond_frames.bin";
    if (!std::filesystem::exists(frameBin))
      throw std::runtime_error(
          "ref2va requires cond_frames.bin (1x3xTxHxW float32)");
    MemRef<float, 5> frames({1, 3, kFrames, kHeight, kWidth});
    std::ifstream in(frameBin, std::ios::binary);
    in.read(reinterpret_cast<char *>(frames.getData()),
            frames.getSize() * sizeof(float));
    if (!in)
      throw std::runtime_error("failed to read cond_frames.bin");
    MemRef<float, 5> condLatent({1, kLatentC, kLatentT, kLatentH, kLatentW});
    _mlir_ciface_forward_visual_vae_encode(&condLatent, &arg0_v, &frames);
    std::memcpy(videoLatent.getData(), condLatent.getData(),
                videoLatent.getSize() * sizeof(float));
  }

  auto timesteps = makeTimesteps(kDenoiseSteps, kVideoShift);
  MemRef<float, 5> videoOut({1, kLatentC, kLatentT, kLatentH, kLatentW});
  MemRef<float, 3> audioOut({1, kAudioLatentC, kAudioLatentT});
  VideoAudio result(videoOut, audioOut);
  MemRef<float, 1> tMem({1});
  for (int step = 0; step < kDenoiseSteps; ++step) {
    tMem.getData()[0] = timesteps[step];
    _mlir_ciface_forward_transformer(&result, &arg0_tr, &videoLatent, &tMem,
                                     &textEmb, &audioLatent);
    // Euler: x <- x + (x1 - x) * dt with velocity = clean - noise prediction.
    float dt = (step + 1 < kDenoiseSteps)
                   ? (timesteps[step + 1] - timesteps[step])
                   : (0.0f - timesteps[step]);
    for (size_t i = 0; i < videoLatent.getSize(); ++i)
      videoLatent.getData()[i] += result.video.getData()[i] * dt;
    for (size_t i = 0; i < audioLatent.getSize(); ++i)
      audioLatent.getData()[i] += result.audio.getData()[i] * dt;
    std::cout << "[denoise] step " << step << "/" << kDenoiseSteps << std::endl;
  }

  MemRef<float, 5> framesOut({1, 3, kFrames, kHeight, kWidth});
  MemRef<float, 3> wavOut({1, 2, kSampleRate * kDurationS});
  _mlir_ciface_forward_visual_vae(&framesOut, &arg0_v, &videoLatent);
  _mlir_ciface_forward_audio_vae(&wavOut, &arg0_a, &audioLatent);

  const std::string mp4_768 = "h3_ref2va_768p.mp4";
  writeRawAndMux(framesOut, wavOut, mp4_768);

  // 3) Regenerate-2K
  const std::string mp4_2k = "h3_ref2va_2k.mp4";
  {
    std::ostringstream cmd;
    cmd << "python3 " << apiPy << " regenerate-2k --prompt-file " << contextPath
        << " --video " << mp4_768 << " --out " << mp4_2k;
    runOrDie(cmd.str());
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "[Output] 768p: " << mp4_768 << std::endl;
  std::cout << "[Output] 2K:   " << mp4_2k << std::endl;
  std::cout << "[Log] elapsed_s=" << sec << std::endl;
  return 0;
}

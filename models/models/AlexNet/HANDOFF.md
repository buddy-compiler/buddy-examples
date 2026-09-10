# HANDOFF — AlexNet (Pie33000/alexnet) ModelTest e2e workload

Model: https://huggingface.co/Pie33000/alexnet — a from-scratch PyTorch
AlexNet trained on ImageNet-1K (5 conv + 3 fc, no LRN, 201,838,952 fp32
params). Original training/validation code:
https://github.com/pie33000/alexnet (`model.py`, `load_data.py`).
Checkpoint used: `model_40.pth` (epoch 40; `top1_accuracy.txt` peaks at
~44.5%, consistent with this model being trained from scratch for only 40
epochs).

This is the **initial model-adaptation stage**: model-side only. No chip
binding, no `archs/buckyball/`, no quant/trace, no rushB/simulation/FPGA, no
buddy-opt/buddy-translate/buddy-llc lowering pipeline is committed here.

## 1. Artifacts

Workload directory: `bb-tests/workloads/src/ModelTest/e2e/models/models/AlexNet/`

| File | Purpose |
| --- | --- |
| `buddy-alexnet-import.py` | AOT importer: rebuilds the exact AlexNet architecture from the author's `model.py`, loads `model_40.pth` (downloaded from HF on first run, gitignored), imports with the buddy DynamoCompiler (TOSA registry) and emits `subgraph0.mlir`, `forward.mlir`, `arg0.data` |
| `buddy-alexnet-main.cpp` | Driver: loads `images/dog-326x256.bmp` via DIP, center-crops 224x224, applies ImageNet mean/std normalization, runs the MLIR forward, prints top-5 labels |
| `CMakeLists.txt` | `add_custom_command` + `alexnet-model-build` target producing the three import artifacts |
| `Labels.txt` | ImageNet-1K labels (identical to ResNet18 workload's) |
| `images/dog-326x256.bmp` / `.png` | Fixed input: ImageNet dog photo (source: `ResNet18/images/dog.png`, 1546x1213) resized so the short side is exactly 256 (326x256), making the `Resize(256) -> CenterCrop(224)` pipeline a pure deterministic center crop |
| `reference/alexnet-reference.npz` | Canonical full-output reference: preprocessed input (1,3,224,224), full 1000-dim logits and softmax |
| `reference/alexnet-reference.txt` | Human-readable dump of the same reference (logits, probs, top-5 with labels) |
| `reference/generate_reference.py` | Reproduces the reference with the author's architecture + checkpoint |
| `.gitignore` | excludes `arg0.data`, `*.mlir`, `model_*.pth` (807 MB checkpoint), `__pycache__` |

Registration (three places):
1. `bb-tests/workloads/src/ModelTest/e2e/models/CMakeLists.txt` — `ALEXNET` added to the `foreach(model_flag IN ITEMS ...)` reset list.
2. `bb-tests/workloads/src/ModelTest/e2e/models/models/CMakeLists.txt` — `set(MODEL_ALEXNET_DIR ...)` and `if(MODEL_ALEXNET) add_subdirectory(AlexNet)`.
3. `bbdev/api/steps/workload/01_build_event.step.py` — new `MODEL_CMAKE` dict with `"alexnet": "alexnet"` (CLI alias -> `-DMODEL` value).

## 2. Canonical Reference

Produced by `reference/generate_reference.py` (torch, fp32, eval mode,
dropout disabled):

```
checkpoint loaded: .../AlexNet/model_40.pth | state dict keys: 16
input: (1, 3, 224, 224) min/max: -2.1007792949676514 2.640000104904175
top-5:
  1: [258] Samoyed        p=0.585141 logit=15.3390
  2: [157] papillon       p=0.198675 logit=14.2588
  3: [257] Great Pyrenees p=0.044844 logit=12.7704
  4: [259] Pomeranian     p=0.037627 logit=12.5949
  5: [222] kuvasz         p=0.033490 logit=12.4784
```

The input is a real dog photo and the model correctly classifies it as
Samoyed (index 258). Full 1000-dim logits/softmax are stored in
`reference/alexnet-reference.npz` (also dumped as text).

## 3. Local Run

Device: **CPU** (x86-64 host, single-threaded MLIR loops; one forward =
~6.1 s < 10 s, so no GPU needed per the stage rule).

Commands (in `nix develop`, `BUCKYBALL_COMPILER_CHIP=toy`):

1. Import (downloads `model_40.pth` on first run):
   ```
   cd bb-tests/workloads/src/ModelTest/e2e/models/models/AlexNet
   python3 buddy-alexnet-import.py --output-dir .
   # [Import] params: 201838952
   # [Import] arg0.data floats: 201838952
   ```
2. Host build (scratch, not committed): lower `subgraph0.mlir`/`forward.mlir`
   with the tosa->linalg->loops pipeline (`buddy-opt`/`buddy-translate`/
   `buddy-llc`, x86-64), compile `buddy-alexnet-main.cpp`, link with
   `CRunnerUtils` + MLIR runner utils.
3. Run:
   ```
   ./buddy-alexnet-run        # cwd contains images/, arg0.data, Labels.txt
   ```
   Output (top-5):
   ```
   Top 1: Index 258, Label "Samoyed", Probability 0.585142
   Top 2: Index 157, Label "papillon", Probability 0.198676
   Top 3: Index 257, Label "Great Pyrenees", Probability 0.044844
   Top 4: Index 259, Label "Pomeranian", Probability 0.037627
   Top 5: Index 222, Label "kuvasz", Probability 0.033491
   ```

Alignment with the reference (MLIR forward on the exact reference input
tensor, compared against `alexnet-reference.npz`):
```
max abs logits diff: 9.536743e-06
mean abs logits diff: 8.526175e-07
torch argmax: 258  mlir argmax: 258
top5 identical: True
```
The full 1000-dim logits agree with torch to ~1e-5 (pure fp32 accumulation
noise), argmax and top-5 are identical. The end-to-end driver path (DIP
decode + center crop + normalize -> MLIR forward) reproduces the same top-5
with probabilities within 1e-5 of the reference.

## 4. Build Binding

Only the model-build custom target is committed (`CMakeLists.txt`); it runs
the importer. Configure/build commands used locally:

```
cmake -G Ninja -S bb-tests/workloads -B <build> \
  -DPython3_EXECUTABLE=<nix python3> -DBUDDY_MLIR_BUILD_DIR=... \
  -DMODEL=alexnet
ninja -C <build> alexnet-model-build
```
(equivalent to what `bb-tests/workloads/scripts/build.py` does for the
workload configure step with `-DMODEL=alexnet`).

For local verification the runnable driver was built by hand (host x86):
subgraph0/forward lowered with `buddy-opt -pass-pipeline
"builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg,
tosa-to-tensor, tosa-to-arith))"` -> `-one-shot-bufferize=...` ->
`-convert-linalg-to-loops` -> LLVM -> `buddy-llc -mtriple=x86_64`; linked
with `libmlir_runner_utils`, `libmlir_c_runner_utils`,
`libmlir_async_runtime` and `CRunnerUtils.o`. This mirrors the per-arch
build in `archs/*/AlexNet/CMakeLists.txt` that a later stage will add.

## 5. Known Limitation

1. **buddy-mlir TOSA max_pool2d lowering bug (worked around in the importer).**
   The TOSA dialect requires `(H + pad_top + pad_bottom - kernel) % stride ==
   0`; for AlexNet's second max-pool (2x2, stride 2 on the 27x27 map) torch's
   floor semantics give 13x13, which TOSA cannot express with any padding
   ((27 + p - 2) % 2 == 0 forces p odd -> 14x14). The upstream importer emits
   a padded 14x14 pool while keeping 13x13 downstream types, which the TOSA
   verifier rejects. `buddy-alexnet-import.py` post-processes the generated
   `subgraph0.mlir` to keep the padded pool and insert a `tosa.slice` back to
   13x13 (same pad+slice trick the importer already uses for conv1). The
   numerical check above (max logits diff ~1e-5) confirms the slice path is
   semantically identical to torch. If the buddy-mlir importer is fixed
   upstream, `fix_maxpool_floor_semantics` should become a no-op and can be
   removed.
2. **Single-crop evaluation.** The author's validation used TenCrop(224)
   (5 crops + mirrors, averaged). For a fixed deterministic workload input we
   use a single CenterCrop(224) of a 326x256 image; Resize(256) is an
   identity on the asset, so runtime preprocessing is interpolation-free and
   fully reproducible.
3. **No chip/arch binding yet** (by design of this stage): no
   `archs/buckyball/`, no `MODEL_LAYOUT` entry, no rushB/simulation/FPGA
   targets, no quant/trace. The 807 MB `model_40.pth` is downloaded at import
   time and gitignored; `arg0.data`/`*.mlir` are build products (gitignored).
4. **Checkpoint accuracy is modest** (~44.5% top-1, 40-epoch from-scratch
   training per the author's log) — classification quality reflects the
   upstream model, not this workload.

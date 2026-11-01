# ===- buddy-pytorch-importer.py ------------------------------------------
#
# SmolVLA on piplus, a multi-core device: the policy as its three classes of
# graphs — the vision encoder (once per camera), the prefix (once) and one
# denoising step (once per step) — and the device program that launches
# them, each graph quantized to the MXINT8 products of piplus; the inputs of
# the program, one real observation (the 100th frame of SO100 episode 40,
# models/smolvla-base/observation.py), and the actions of the quantized program in PyTorch on
# it, the reference of the device.
#
# ===---------------------------------------------------------------------===

from importlib import import_module
from pathlib import Path
import json

import numpy as np
import torch
from torch._inductor.decomposition import decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from compiler.frontend import quant
from frontend import program
from frontend.graphs import Prefix, Step, Vision
from frontend.quant.graph import doubled, quantize
from frontend.quant.layout import lay_out
from frontend.quant.reference import Products, mxint8

model_dir = Path(__file__).resolve().parent
model = import_module("models.models.smolvla-base.model")
observation = import_module("models.models.smolvla-base.observation")
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)

policy = model.load()
# The vision encoder attends through explicit products the quantization can
# read as MXINT8, not through one fused attention.
policy.flow.vlm_with_expert.get_vlm_model().vision_model.config._attn_implementation = (
    "eager"
)
flow = policy.flow
images, lang_tokens, lang_masks, state, noise = observation.frame(40)
steps = flow.config.num_steps

# ===----------------------------------------------------------------------===
# Step 1: Cut the policy into its graphs, check that the program of them
# computes what the policy does, and import each.
# ===----------------------------------------------------------------------===
vision, prefix, step = Vision(flow).eval(), Prefix(flow).eval(), Step(flow).eval()
with torch.no_grad():
    tokens = [vision(image) for image in images.unbind(0)]
    pad_masks, *cache = prefix(lang_tokens, lang_masks, state, *tokens)
    x_t = program.run(
        vision, prefix, step, steps, images, lang_tokens, lang_masks, state, noise
    )
    actions = policy(images, lang_tokens, lang_masks, state, noise)
print(
    f"Program of graphs vs policy: max |difference| {(x_t - actions).abs().max().item()}"
)

# A compiler keeps the graphs it imported: one compiler per graph.
graphs = {}
params = {}
examples = {}
for name, module, example in (
    ("vision", vision, (images[0],)),
    ("prefix", prefix, (lang_tokens, lang_masks, state, *tokens)),
    ("step", step, (noise, torch.ones(1), pad_masks, *cache)),
):
    compiler = DynamoCompiler(
        primary_registry={**tosa.ops_registry, **quant.ops_registry},
        aot_autograd_decomposition=decompositions,
        trace=None,
    )
    with torch.no_grad():
        (graphs[name],) = compiler.importer(module, *example)
    params[name] = compiler.imported_params[graphs[name]]
    examples[name] = example

# ===----------------------------------------------------------------------===
# Step 2.1: Export the graphs as imported.
# ===----------------------------------------------------------------------===
for name, graph in graphs.items():
    graph.dump(output / f"01-import-{name}.txt")

# ===----------------------------------------------------------------------===
# Step 2.3: Quantize every matrix product of each graph to MXINT8, K
# zero-padded to a multiple of 32 where it is not one; the weights of the
# vision MLP as two MXINT8 terms.
# ===----------------------------------------------------------------------===
# The activation scales are computed online; buddy-pytorch-calibrate.py
# accepts the quantized program of smolvla-base first. Its actions on the
# observation are the reference of the device.
calibration = json.loads((output / "calibration.json").read_text())
print(f"Quantized actions accepted at relative L2 error {calibration['error']:.4f}")
with torch.no_grad(), Products(mxint8, doubled(flow)):
    reference = program.run(
        vision, prefix, step, steps, images, lang_tokens, lang_masks, state, noise
    )
    # One denoising step alone: the first, on the prefix of the quantized
    # program, and its velocity.
    tokens = [vision(image) for image in images.unbind(0)]
    pad_masks, *cache = prefix(lang_tokens, lang_masks, state, *tokens)
    time = torch.ones(1)
    velocity = step(noise, time, pad_masks, *cache)
for name, value in (
    ("step_time", time),
    ("step_pad_masks", pad_masks),
    ("step_cache", torch.cat([c.reshape(-1) for c in cache])),
    ("step_velocity", velocity),
    ("images", images),
    ("lang_tokens", lang_tokens),
    ("lang_masks", lang_masks),
    ("state", state),
    ("noise", noise),
    ("actions", reference),
):
    value.numpy().tofile(output / f"{name}.data")
for name, graph in graphs.items():
    params[name] = quantize(graph, params[name], doubled(flow))
    # The weights in the layout of the mxint8mm Ball, read in place.
    params[name] = lay_out(graph, params[name])
    graph.dump(output / f"03-quanted-{name}.txt")
    # A graph takes its parameters grouped by type: FP32 (the biases among
    # them), INT8 (the MXINT8 codes and exponents of the weights).
    for index, dtype in enumerate((torch.float32, torch.int8)):
        np.concatenate(
            [
                p.detach().cpu().numpy().reshape(-1)
                for p in params[name]
                if p.dtype == dtype
            ]
        ).tofile(output / f"{name}-arg{index}.data")

# ===----------------------------------------------------------------------===
# Step 3: Export every graph as an MLIR module, graph-<name>.mlir, the
# device program that launches them, program.mlir, and the program of one
# denoising step alone, step_program.mlir.
# ===----------------------------------------------------------------------===
signatures = {}
for name, graph in graphs.items():
    text, signatures[name] = program.graph(name, graph, examples[name])
    (output / f"graph-{name}.mlir").write_text(text)
(output / "program.mlir").write_text(
    program.program(signatures, steps, images.shape[0])
)
(output / "step_program.mlir").write_text(program.step_program(signatures))

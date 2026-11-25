# ===- buddy-pytorch-importer.py ------------------------------------------
#
# pi0.5 on piplus, a multi-hart device: the policy as its three classes of
# graphs — the vision encoder (once per camera), the prefix (once) and one
# denoising step (once per step) — and the device program that launches
# them, each graph quantized to the MXINT8 products of piplus, and the program
# of one denoising step alone. The graphs are traced on one real observation
# (the 100th frame of SO100 episode 40, models/pi05-base/observation.py).
#
# ===---------------------------------------------------------------------===

from importlib import import_module
from pathlib import Path
import sys
import json

import numpy as np
import torch
from torch._inductor.decomposition import decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from compiler.frontend import quant
from frontend import graphs as recipe
from frontend import program
from frontend.graphs import Prefix, Step, Vision
from transformers.models.gemma import modeling_gemma
from frontend.partition.graph import cut
from frontend.partition.layout import ball
from frontend.quant.graph import doubled, quantize
from frontend.quant.layout import lay_out
from scripts.design import load

model_dir = Path(__file__).resolve().parent
model = import_module("models.models.pi05-base.model")
observation = import_module("models.models.pi05-base.observation")
output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
design = load(sys.argv[2])

policy = model.load()
flow = policy.flow
# The vision encoder attends through explicit products the quantization can
# read as MXINT8, not through one fused attention.
flow.paligemma_with_expert.paligemma.model.vision_tower.config._attn_implementation = (
    "eager"
)
images, tokens, token_masks, noise = observation.frame(40)
steps = flow.config.num_inference_steps

# ===----------------------------------------------------------------------===
# Step 1: Cut the policy into its graphs and import each.
# ===----------------------------------------------------------------------===
vision, prefix = Vision(flow).eval(), Prefix(flow).eval()
step = Step(flow, model.PrefixCache).eval()
with torch.no_grad():
    reference = policy(images, tokens, token_masks, noise)
    # The graphs attend with the bias before the scale and encode their
    # positions without a matrix product (graphs.rewrite).
    recipe.rewrite(
        flow.paligemma_with_expert.paligemma.model.language_model,
        step.flow.paligemma_with_expert.gemma_expert.model,
    )
    image_tokens = [vision(image) for image in images.unbind(0)]
    # The vision embeddings in token order (graphs.Vision) are the policy's.
    difference = max(
        (token - flow.paligemma_with_expert.embed_image(image)).abs().max()
        for token, image in zip(image_tokens, images.unbind(0))
    )
    print(
        f"Vision in token order against the policy's: max |difference| {difference:.3e}"
    )
    assert difference < 1e-3
    bias, positions, *cache = prefix(tokens, token_masks, *image_tokens)
    actions = program.run(
        vision, prefix, step, steps, images, tokens, token_masks, noise
    )
    difference = (actions - reference).abs().max()
    print(f"The graphs against the policy: max |difference| {difference:.3e}")
    assert difference < 1e-4

# A compiler keeps the graphs it imported: one compiler per graph.
graphs = {}
params = {}
examples = {}
for name, module, example in (
    ("vision", vision, (images[0],)),
    ("prefix", prefix, (tokens, token_masks, *image_tokens)),
    ("step", step, (noise, torch.ones(1), bias, positions, *cache)),
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
# zero-padded to a multiple of 32 where it is not one;
# Linear weights as two MXINT8 terms.
# ===----------------------------------------------------------------------===
# The activation scales are computed online.
devices, device_inputs = {}, {}
for name, graph in graphs.items():
    # The weights of the policy and those of the widened SigLIP (graphs.Vision).
    params[name] = quantize(graph, params[name], doubled(flow) | doubled(vision))
    if name == "vision":
        # The weights in the layout of the mxint8mm Ball, read in place.
        params[name] = lay_out(graph, params[name])
        graph.dump(output / f"03-quanted-{name}.txt")
        # A graph takes its parameters grouped by type: FP32 (the biases
        # among them), INT8 (the MXINT8 codes and exponents of the weights).
        for index, dtype in enumerate((torch.float32, torch.int8)):
            np.concatenate(
                [
                    p.detach().cpu().numpy().reshape(-1)
                    for p in params[name]
                    if p.dtype == dtype
                ]
            ).tofile(output / f"{name}-arg{index}.data")
        continue
    # =====================================================================
    # Step 2.4: Partition the prefix and the step across the harts: each
    # rank its query heads of the one key head (whose keys and values every
    # rank computes) and its MLP channels, every hart its own copy of what
    # the masks and positions give.
    # =====================================================================
    gemma = flow.paligemma_with_expert.gemma_expert.model.config
    device = cut(
        graph,
        params[name],
        design,
        name,
        examples[name],
        {1} if name == "prefix" else {2, 3},
        gemma.num_key_value_heads,
        gemma.num_attention_heads // gemma.num_key_value_heads,
        gemma.head_dim,
    )
    ball(device)
    (output / f"04-partitioned-{name}.txt").write_text(
        "".join(
            f"{s.name}: graph {s.graph} (on {s.hart_class} hart {s.hart}), {len(s.ops)} operations\n"
            f"  inputs: {', '.join(s.inputs)}\n  outputs: {', '.join(s.outputs)}\n"
            for s in device.segments
        )
    )
    # The program takes the inputs as the graph does.
    order = [
        next(i for i, value in enumerate(examples[name]) if value is input_)
        for input_ in graph._runtime_inputs_ref
    ]
    inputs = [
        node.name
        for _, node in sorted(zip(order, graph.inputs), key=lambda item: item[0])
    ]
    if name == "step":
        (output / "step_program.mlir").write_text(
            device.program("step_program", ["f32", "i8"], inputs)
        )
    for hart_class in ("action", "dense"):
        (output / f"graph-{name}-{hart_class}.mlir").write_text(
            device.module(hart_class)
        )
    for element, values_ in device.packed().items():
        values_.tofile(output / f"{name}-pack-{element}.data")
    devices[name], device_inputs[name] = device, inputs
    graph.dump(output / f"03-quanted-{name}.txt")

# ===----------------------------------------------------------------------===
# Step 3: Export the vision graph as an MLIR module, graph-vision.mlir, and
# the device program that launches it, the partitioned prefix and the
# partitioned steps, program.mlir.
# ===----------------------------------------------------------------------===
text, signature = program.graph("vision", graphs["vision"], examples["vision"])
(output / "graph-vision.mlir").write_text(text)
(output / "program.mlir").write_text(
    program.program(
        signature,
        steps,
        images.shape[0],
        devices["prefix"],
        device_inputs["prefix"],
        devices["step"],
        device_inputs["step"],
    )
)

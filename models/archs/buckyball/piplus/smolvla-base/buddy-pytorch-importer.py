# ===- buddy-pytorch-importer.py ------------------------------------------
#
# SmolVLA on piplus, a multi-hart device: the policy as its three classes of
# graphs — the vision encoder (once per camera), the prefix (once) and one
# denoising step (once per step) — and the device program that launches
# them, each graph quantized to the MXINT8 products of piplus, and the program
# of one denoising step alone. The graphs are traced on one real observation
# (the 100th frame of SO100 episode 40, models/smolvla-base/observation.py);
# buddy-pytorch-golden.py gives the inputs and outputs the device is checked
# with.
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
from frontend import program
from frontend.graphs import Prefix, Step, Vision
from frontend.partition.graph import cut
from frontend.partition.layout import ball
from frontend.quant.graph import doubled, quantize
from frontend.quant.layout import lay_out
from scripts.design import load

model_dir = Path(__file__).resolve().parent
model = import_module("models.models.smolvla-base.model")
observation = import_module("models.models.smolvla-base.observation")
output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
design = load(sys.argv[2])

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
# Step 1: Cut the policy into its graphs and import each.
# ===----------------------------------------------------------------------===
vision, prefix, step = Vision(flow).eval(), Prefix(flow).eval(), Step(flow).eval()
with torch.no_grad():
    tokens = [vision(image) for image in images.unbind(0)]
    # The vision embeddings in token order (graphs.Vision) are the policy's.
    difference = max(
        (token - flow.vlm_with_expert.embed_image(image)).abs().max()
        for token, image in zip(tokens, images.unbind(0))
    )
    print(
        f"Vision in token order against the policy's: max |difference| {difference:.3e}"
    )
    assert difference < 1e-3
    masks, angles, *cache = prefix(lang_tokens, lang_masks, state, *tokens)
    # The prefix with RoPE over whole rows (frontend/rope.py) is the policy's.
    vlm = prefix.flow.vlm_with_expert
    layers = vlm.forward_attn_layer, vlm.forward_cross_attn_layer
    encode = prefix.encode
    del vlm.forward_attn_layer, vlm.forward_cross_attn_layer
    prefix.encode = lambda positions: positions
    _, _, *policy = prefix.raw(lang_tokens, lang_masks, state, *tokens)
    vlm.forward_attn_layer, vlm.forward_cross_attn_layer = layers
    prefix.encode = encode
    _, _, *raw = prefix.raw(lang_tokens, lang_masks, state, *tokens)
    difference = max((a - b).abs().max() for a, b in zip(raw, policy))
    print(
        f"Prefix with RoPE over rows against the policy's: max |difference| {difference:.3e}"
    )
    assert difference < 1e-3
    # The padded step (its chunk and its hidden, the prefix's masks,
    # angles and projected cross keys and values) is the policy's step on
    # the language model's keys and values; the prefix's padding is the first
    # row of its masks.
    pad_masks = masks[:, 0, : raw[0].shape[1]]
    half = len(raw) // 2
    past = {
        layer: {"key_states": raw[layer], "value_states": raw[half + layer]}
        for layer in range(half)
    }
    difference = (
        (
            step(noise, step.time_bias(torch.ones(1)), masks, angles, *cache)
            - flow.denoise_step(pad_masks, past, noise, torch.ones(1))
        )
        .abs()
        .max()
    )
    print(f"Padded step against the policy's: max |difference| {difference:.3e}")
    assert difference < 1e-4

# A compiler keeps the graphs it imported: one compiler per graph.
graphs = {}
params = {}
examples = {}
for name, module, example in (
    ("vision", vision, (images[0],)),
    ("prefix", prefix, (lang_tokens, lang_masks, state, *tokens)),
    ("step", step, (noise, step.time_bias(torch.ones(1)), masks, angles, *cache)),
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
# The activation scales are computed online; buddy-pytorch-calibrate.py
# accepts the quantized program of smolvla-base first.
calibration = json.loads((model_dir / "output/calibration.json").read_text())
print(f"Quantized actions accepted at relative L2 error {calibration['error']:.4f}")
devices, device_inputs = {}, {}
for name, graph in graphs.items():
    params[name] = quantize(graph, params[name], doubled(flow))
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
    if name == "prefix":
        # =================================================================
        # Step 2.4: Partition the prefix across the harts the same way:
        # each rank the query heads of its key heads and its MLP channels,
        # every hart its own copy of what the masks and positions give
        # (lang_masks, input 1) and of the projection of the state (input
        # 2), a token too small to split.
        # =================================================================
        text = flow.vlm_with_expert.get_vlm_model().text_model
        device = cut(
            graph,
            params[name],
            design,
            "prefix",
            examples[name],
            {1, 2},
            text.config.num_key_value_heads,
            text.config.num_attention_heads // text.config.num_key_value_heads,
            text.layers[0].self_attn.head_dim,
        )
    if name == "step":
        # =================================================================
        # Step 2.4: Partition the step across the harts as the program of
        # one denoising step: launches of its semantic graphs, each defined
        # once per class of hart (graph-step-<class>.mlir), the weights of
        # each rank in the layout of the mxint8mm Ball.
        # =================================================================
        expert = flow.vlm_with_expert.lm_expert
        device = cut(
            graph,
            params[name],
            design,
            "step",
            examples[name],
            {2, 3},
            expert.config.num_key_value_heads,
            expert.config.num_attention_heads // expert.config.num_key_value_heads,
            expert.layers[0].self_attn.head_dim,
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
    hart_class = {"prefix": "dense", "step": "action"}[name]
    (output / f"graph-{name}-{hart_class}.mlir").write_text(device.module(hart_class))
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
        [step.time_bias(torch.full((1,), 1.0 - k / steps)) for k in range(steps)],
    )
)

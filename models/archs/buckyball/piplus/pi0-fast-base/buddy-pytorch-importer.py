# ===- buddy-pytorch-importer.py ------------------------------------------
#
# pi0-FAST on piplus, a multi-hart device: the policy as its three classes of
# graphs — the vision encoder (once per camera), the prefill (once) and one
# decoding step (once per token) — each quantized to the MXINT8 products of
# piplus, and its two device programs: the prefill (the cameras, then the
# prefill graph) and one decoding step, which the runner launches until the
# end of the sequence. The graphs are traced on one real observation (the
# 100th frame of SO100 episode 40, models/pi0-fast-base/observation.py).
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
from frontend.graphs import Decode, Prefill, Vision, generated
from frontend.partition.graph import cut
from frontend.partition.layout import ball
from frontend.quant.graph import doubled, quantize
from frontend.quant.layout import lay_out
from scripts.design import load

model_dir = Path(__file__).resolve().parent
model = import_module("models.models.pi0-fast-base.model")
observation = import_module("models.models.pi0-fast-base.observation")
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
images, tokens, token_masks = observation.frame(40)
eos = observation.tokenizer.eos_token_id

# ===----------------------------------------------------------------------===
# Step 1: Cut the policy into its graphs and import each.
# ===----------------------------------------------------------------------===
vision, prefix, step = Vision(flow).eval(), Prefill(flow).eval(), Decode(flow).eval()
with torch.no_grad():
    reference = policy(images, tokens, token_masks)[0].tolist()
    reference = reference[: reference.index(eos) + 1]
    # The graphs attend with the bias before the scale and encode their
    # positions without a matrix product (graphs.rewrite).
    recipe.rewrite(
        flow.paligemma_with_expert.paligemma.model.language_model,
        step.flow.paligemma_with_expert.paligemma.model.language_model,
    )
    # The prompt with the token the model continues, as the runner frames it.
    tokens, token_masks = program.frame(flow, tokens, token_masks)
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
    produced, _ = program.run(
        flow, vision, prefix, step, eos, images, tokens, token_masks
    )
    assert produced == reference
    print(
        f"The graphs against the policy: the {len(produced)} tokens up to the EOS agree"
    )
    # The first decoding step: the first token, no generated key yet.
    prompt = program.rows(flow, tokens)
    logits, position, bias, *cache = prefix(prompt, token_masks, *image_tokens)
    token = program.rows(flow, logits[:, -1].argmax(-1, keepdim=True))
    before = [torch.zeros(1, 1, generated, cache[0].shape[3]) for _ in cache]

# A compiler keeps the graphs it imported: one compiler per graph.
graphs = {}
params = {}
examples = {}
for name, module, example in (
    ("vision", vision, (images[0],)),
    ("prefix", prefix, (prompt, token_masks, *image_tokens)),
    ("step", step, (token, position, bias, *cache, *before)),
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
    # Step 2.4: Partition the prefill and the step across the harts: each
    # rank its query heads of the one key head (whose keys and values every
    # rank computes), its MLP channels and its tokens of the vocabulary,
    # every hart its own copy of what the masks and positions give.
    # =====================================================================
    gemma = flow.paligemma_with_expert.paligemma.model.language_model.config
    device = cut(
        graph,
        params[name],
        design,
        name,
        examples[name],
        {1} if name == "prefix" else {1, 2},
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
        (output / "decode_program.mlir").write_text(
            device.program("decode_program", ["f32", "i8"], inputs)
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
# Step 3: Export the vision graph as an MLIR module, graph-vision.mlir, the
# device program that launches it for every camera and then the partitioned
# prefill, prefill_program.mlir, and the embedding table the runner reads the
# rows of the tokens from, embedding.data.
# ===----------------------------------------------------------------------===
text, signature = program.graph("vision", graphs["vision"], examples["vision"])
(output / "graph-vision.mlir").write_text(text)
(output / "prefill_program.mlir").write_text(
    program.prefill(
        signature, images.shape[0], devices["prefix"], device_inputs["prefix"]
    )
)
program.rows(flow, slice(None)).detach().numpy().tofile(output / "embedding.data")

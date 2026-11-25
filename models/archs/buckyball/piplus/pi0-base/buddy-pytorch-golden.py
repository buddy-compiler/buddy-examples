# ===- buddy-pytorch-golden.py --------------------------------------------
#
# pi0 on piplus: the golden data of baremetal-step.cpp. One real
# observation (the 100th frame of SO100 episode 40,
# models/pi0-base/observation.py); the first denoising step alone, its
# inputs from the quantized program in PyTorch on the observation
# (frontend/quant/reference.py) and its velocity; and the actions of the
# quantized program.
#
#   buddy-pytorch-golden.py <libnewlib-reference.so> <output>   the newlib
#                                       library of the harts' tanhf
#
# ===---------------------------------------------------------------------===

from importlib import import_module
from pathlib import Path
import sys

import torch
from transformers.models.gemma import modeling_gemma

from frontend import graphs as recipe
from frontend import program
from frontend.graphs import Prefix, Step, Vision
from frontend.quant.graph import doubled
from frontend.quant.reference import Products, mxint8

newlib = sys.argv[1]
model = import_module("models.models.pi0-base.model")
observation = import_module("models.models.pi0-base.observation")
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)

policy = model.load()
flow = policy.flow
# The vision encoder attends through explicit products, as the importer
# imports it.
flow.paligemma_with_expert.paligemma.model.vision_tower.config._attn_implementation = (
    "eager"
)
images, tokens, token_masks, state, noise = observation.frame(40)
steps = flow.config.num_inference_steps

vision, prefix = Vision(flow).eval(), Prefix(flow).eval()
step = Step(flow, model.PrefixCache).eval()
with torch.no_grad():
    actions = policy(images, tokens, token_masks, state, noise)
    # The graphs attend with the bias before the scale and encode their
    # positions without a matrix product (graphs.rewrite).
    recipe.rewrite(
        flow.paligemma_with_expert.paligemma.model.language_model,
        step.flow.paligemma_with_expert.gemma_expert.model,
    )
    x_t = program.run(
        vision, prefix, step, steps, images, tokens, token_masks, state, noise
    )
print(
    f"Program of graphs vs policy: max |difference| {(x_t - actions).abs().max().item()}"
)

# The quantized program, as the quantized graphs compute it: its actions,
# and one denoising step alone: the first, on the prefix of the quantized
# program, and its velocity.
two_terms = doubled(flow) | doubled(vision)
with torch.no_grad():
    # A camera's vision graph runs whole on one dense hart; the prefix and
    # the steps are cut over the harts.
    with Products(mxint8, two_terms, newlib, True):
        image_tokens = [vision(image) for image in images.unbind(0)]
    with Products(mxint8, two_terms, newlib, False):
        cameras = iter(image_tokens)
        reference = program.run(
            lambda image: next(cameras),
            prefix,
            step,
            steps,
            images,
            tokens,
            token_masks,
            state,
            noise,
        )
        bias, positions, *cache = prefix(tokens, token_masks, *image_tokens)
        time = torch.ones(1)
        velocity = step(noise, time, state, bias, positions, *cache)
error = ((reference - actions).norm() / actions.norm()).item()
print(
    f"Quantized program vs policy: max |difference| {(reference - actions).abs().max().item()}, "
    f"relative L2 error {error:.4f}"
)
for name, value in (
    ("step_time", time),
    ("step_state", state),
    ("step_bias", bias),
    ("step_positions", positions),
    ("step_cache", torch.cat([c.reshape(-1) for c in cache])),
    ("step_velocity", velocity),
    ("images", images),
    ("tokens", tokens),
    ("token_masks", token_masks),
    ("state", state),
    ("noise", noise),
    ("actions", reference),
):
    value.numpy().tofile(output / f"{name}.data")
(output / "observation.data").write_bytes(
    b"".join(
        value.numpy().tobytes() for value in (images, tokens, token_masks, state, noise)
    )
)

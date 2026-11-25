# ===- buddy-pytorch-golden.py --------------------------------------------
#
# SmolVLA on piplus: the golden data of baremetal.cpp and baremetal-step.cpp.
# The inputs of the program, one real observation (the 100th frame of SO100
# episode 40, models/smolvla-base/observation.py), and the actions of the
# FP32 program of the graphs on it; for the first denoising step alone, its
# inputs from that program on the observation and its velocity.
#
#   buddy-pytorch-golden.py <libnewlib-reference.so>   the newlib library of
#                                                      the harts' tanhf
#
# ===---------------------------------------------------------------------===

from importlib import import_module
from pathlib import Path
import sys

import torch

from frontend import program
from frontend.graphs import Prefix, Step, Vision

model_dir = Path(__file__).resolve().parent
newlib = sys.argv[1]
model = import_module("models.models.smolvla-base.model")
observation = import_module("models.models.smolvla-base.observation")
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)

policy = model.load()
# The vision encoder attends through explicit products, as the importer
# imports it.
policy.flow.vlm_with_expert.get_vlm_model().vision_model.config._attn_implementation = (
    "eager"
)
flow = policy.flow
images, lang_tokens, lang_masks, state, noise = observation.frame(40)
steps = flow.config.num_steps

# The program of the graphs the importer imports computes what the policy
# does.
vision, prefix, step = Vision(flow).eval(), Prefix(flow).eval(), Step(flow).eval()
with torch.no_grad():
    x_t = program.run(
        vision, prefix, step, steps, images, lang_tokens, lang_masks, state, noise
    )
    actions = policy(images, lang_tokens, lang_masks, state, noise)
print(
    f"Program of graphs vs policy: max |difference| {(x_t - actions).abs().max().item()}"
)

# The reference of the DSE runs: the FP32 program (the DSE measures latency
# and area, not accuracy, and the quantized program in PyTorch takes over an
# hour): its actions, and one denoising step alone, the first, on the prefix
# of the program, and its velocity; the step takes the time term of t = 1.
with torch.no_grad():
    reference = x_t
    tokens = [vision(image) for image in images.unbind(0)]
    masks, angles, *cache = prefix(lang_tokens, lang_masks, state, *tokens)
    time = step.time_bias(torch.ones(1))
    velocity = step(noise, time, masks, angles, *cache)
for name, value in (
    ("step_time", time),
    ("step_masks", masks),
    ("step_angles", angles),
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
(output / "observation.data").write_bytes(
    b"".join(
        value.numpy().tobytes()
        for value in (images, lang_tokens, lang_masks, state, noise)
    )
)

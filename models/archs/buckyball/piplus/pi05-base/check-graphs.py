# The three graphs of pi0.5 on piplus against the policy, in PyTorch.
from importlib import import_module
import sys
import torch

sys.path.insert(0, ".")
from frontend import graphs
from frontend.graphs import Prefix, Step, Vision
from transformers.models.gemma import modeling_gemma

model = import_module("models.models.pi05-base.model")
observation = import_module("models.models.pi05-base.observation")
policy = model.load()
flow = policy.flow
flow.paligemma_with_expert.paligemma.model.vision_tower.config._attn_implementation = (
    "eager"
)
images, tokens, masks, noise = observation.frame(40)
vision, prefix = Vision(flow).eval(), Prefix(flow).eval()
step = Step(flow, model.PrefixCache).eval()
with torch.no_grad():
    reference = policy(images, tokens, masks, noise)
    # The graphs attend with the bias before the scale and encode their
    # positions without a matrix product (graphs.rewrite).
    graphs.rewrite(
        flow.paligemma_with_expert.paligemma.model.language_model,
        step.flow.paligemma_with_expert.gemma_expert.model,
    )
    # The graphs attend with the bias before the scale and encode their
    # positions without a matrix product (graphs.rewrite).
    image_tokens = [vision(image) for image in images.unbind(0)]
    print("image tokens", image_tokens[0].shape)
    bias, positions, *cache = prefix(tokens, masks, *image_tokens)
    print(
        "bias",
        bias.shape,
        "positions",
        positions.shape,
        "layers",
        len(cache) // 2,
        "keys",
        cache[0].shape,
    )
    x_t = noise
    steps = flow.config.num_inference_steps
    for k in range(steps):
        time = torch.full((1,), 1.0 - k / steps)
        x_t = x_t + step(x_t, time, bias, positions, *cache) * (-1.0 / steps)
    print(
        f"Graphs against the policy: max |difference| {(x_t - reference).abs().max():.3e}, max |action| {reference.abs().max():.3f}"
    )

# ===- buddy-pytorch-golden.py --------------------------------------------
#
# pi0-FAST on piplus: the golden data of baremetal.cpp. One real observation
# (the 100th frame of SO100 episode 40, models/pi0-fast-base/observation.py)
# and the tokens the quantized program decodes from it in PyTorch
# (frontend/quant/reference.py), up to and with the EOS.
#
#   buddy-pytorch-golden.py <libnewlib-reference.so> <output>   the newlib
#                                       library of the harts' tanhf
#
# ===---------------------------------------------------------------------===

from importlib import import_module
from pathlib import Path
import sys

import torch

from frontend import graphs as recipe
from frontend import program
from frontend.graphs import Decode, Prefill, Vision, generated
from frontend.quant.graph import doubled
from frontend.quant.reference import Products, mxint8

newlib = sys.argv[1]
model = import_module("models.models.pi0-fast-base.model")
observation = import_module("models.models.pi0-fast-base.observation")
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)

policy = model.load()
flow = policy.flow
# The vision encoder attends through explicit products, as the importer
# imports it.
flow.paligemma_with_expert.paligemma.model.vision_tower.config._attn_implementation = (
    "eager"
)
images, tokens, token_masks = observation.frame(40)
eos = observation.tokenizer.eos_token_id

vision, prefill, decode = Vision(flow).eval(), Prefill(flow).eval(), Decode(flow).eval()
with torch.no_grad():
    policy_tokens = policy(images, tokens, token_masks)[0].tolist()
    policy_tokens = policy_tokens[: policy_tokens.index(eos) + 1]
    # The graphs attend with the bias before the scale and encode their
    # positions without a matrix product (graphs.rewrite).
    recipe.rewrite(
        flow.paligemma_with_expert.paligemma.model.language_model,
        decode.flow.paligemma_with_expert.paligemma.model.language_model,
    )
    # The prompt with the token the model continues, as the runner frames it.
    tokens, token_masks = program.frame(flow, tokens, token_masks)
    produced, _ = program.run(
        flow, vision, prefill, decode, eos, images, tokens, token_masks
    )
print(
    f"Program of graphs vs policy: {len(produced)} tokens, the same: {produced == policy_tokens}"
)

# The quantized program, as the quantized graphs compute it: its tokens.
two_terms = doubled(flow) | doubled(vision)
with torch.no_grad():
    # A camera's vision graph runs whole on one dense hart; the prefill and
    # the decoding are cut over the harts.
    with Products(mxint8, two_terms, newlib, True):
        image_tokens = [vision(image) for image in images.unbind(0)]
    with Products(mxint8, two_terms, newlib, False):
        cameras = iter(image_tokens)
        reference, _ = program.run(
            flow,
            lambda image: next(cameras),
            prefill,
            decode,
            eos,
            images,
            tokens,
            token_masks,
        )
same = sum(a == b for a, b in zip(reference, policy_tokens))
print(
    f"Quantized program vs policy: {len(reference)} tokens against {len(policy_tokens)}, "
    f"{same} the same in place"
)
# The tokens up to and with the EOS, then zeros (the padding token).
padded = torch.zeros(generated, dtype=torch.int64)
padded[: len(reference)] = torch.tensor(reference)
for name, value in (
    ("images", images),
    ("tokens", tokens),
    ("token_masks", token_masks),
    ("generated", padded),
):
    value.numpy().tofile(output / f"{name}.data")
(output / "observation.data").write_bytes(
    b"".join(value.numpy().tobytes() for value in (images, tokens, token_masks))
)

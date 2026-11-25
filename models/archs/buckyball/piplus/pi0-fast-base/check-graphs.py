# The three graphs of pi0-FAST on piplus against the policy, in PyTorch.
from importlib import import_module
import sys
import torch

sys.path.insert(0, ".")
from frontend import graphs, program
from frontend.graphs import Decode, Prefill, Vision

model = import_module("models.models.pi0-fast-base.model")
observation = import_module("models.models.pi0-fast-base.observation")
policy = model.load()
flow = policy.flow
flow.paligemma_with_expert.paligemma.model.vision_tower.config._attn_implementation = (
    "eager"
)
images, tokens, masks = observation.frame(40)
eos = observation.tokenizer.eos_token_id
vision, prefill, decode = Vision(flow).eval(), Prefill(flow).eval(), Decode(flow).eval()
language = flow.paligemma_with_expert.paligemma.model.language_model
with torch.no_grad():
    reference = policy(images, tokens, masks)[0].tolist()
    reference = reference[: reference.index(eos) + 1]
    # The graphs attend with the bias before the scale and encode their
    # positions without a matrix product (graphs.rewrite).
    graphs.rewrite(
        language, decode.flow.paligemma_with_expert.paligemma.model.language_model
    )
    produced, _ = program.run(
        flow, vision, prefill, decode, eos, images, *program.frame(flow, tokens, masks)
    )
print("policy", len(reference), reference[:8])
print("graphs", len(produced), produced[:8])
assert produced == reference
print(f"Graphs against the policy: the {len(produced)} tokens up to the EOS agree")

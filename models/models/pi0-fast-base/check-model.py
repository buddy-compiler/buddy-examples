# pi0-FAST: the model loads and decodes the tokens of one observation.
from importlib import import_module
import time
import torch

model = import_module("models.models.pi0-fast-base.model")
observation = import_module("models.models.pi0-fast-base.observation")
policy = model.load()
images, tokens, masks = observation.frame(40)
start = time.time()
with torch.no_grad():
    generated = policy(images, tokens, masks)
print("prompt ids", tokens[0, :6].tolist(), "valid", int(masks.sum()))
print(
    "tokens",
    tuple(generated.shape),
    generated[0, :24].tolist(),
    f"{time.time() - start:.0f}s",
)
# The FAST tokens up to the EOS, as the ids of the FAST vocabulary.
ids = generated[0].tolist()
end = ids.index(observation.tokenizer.eos_token_id)
vocab = policy.flow.paligemma_with_expert.paligemma.lm_head.weight.shape[0]
skip = policy.flow.config.fast_skip_tokens
print("FAST tokens", end, [vocab - 1 - skip - i for i in ids[:end]])
assert end > 0 and all(0 <= vocab - 1 - skip - i < 2048 for i in ids[:end])

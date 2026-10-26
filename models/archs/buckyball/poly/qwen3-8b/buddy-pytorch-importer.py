from pathlib import Path

import torch
from torch._inductor.decomposition import decompositions
from transformers import AutoModelForCausalLM

from buddy.compiler.frontend import DynamoCompiler
from stack.compiler.frontend.ops import tosa
from frontend.permute.graph import permute

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
model_source = model_dir.parents[3] / "models/qwen3-8b"
# The KV cache holds 256 positions.
capacity = 256


# ===----------------------------------------------------------------------===
# The interface of the graph: tokens written from `start`, and the KV cache.
# ===----------------------------------------------------------------------===
class Cache:
    """The KV cache of the graph inputs: each layer writes its new entries
    from `start` and returns the whole updated cache."""

    def __init__(self, start, keys, values):
        self.start = start
        self.keys = list(keys)
        self.values = list(values)

    def update(self, key, value, layer):
        positions = self.start + torch.arange(key.shape[2])
        self.keys[layer] = self.keys[layer].index_copy(2, positions, key)
        self.values[layer] = self.values[layer].index_copy(2, positions, value)
        return self.keys[layer], self.values[layer]


class Decoder(torch.nn.Module):
    def __init__(self, model, capacity):
        super().__init__()
        self.model = model
        self.capacity = capacity

    def forward(self, tokens, start, keys, values):
        positions = start + torch.arange(tokens.shape[1])
        visible = torch.arange(self.capacity)[None, :] <= positions[:, None]
        mask = torch.where(visible, 0.0, torch.finfo(torch.float32).min)
        cache = Cache(start, keys, values)
        logits = self.model(
            input_ids=tokens,
            position_ids=positions[None, :],
            attention_mask={"full_attention": mask[None, None, :, :]},
            past_key_values=cache,
            use_cache=True,
        ).logits
        return logits, cache.keys, cache.values


# ===----------------------------------------------------------------------===
# Step 1: Import one graph of the checkpoint for any number of tokens.
# ===----------------------------------------------------------------------===
model = AutoModelForCausalLM.from_pretrained(
    model_source / "output/qwen3-8b",
    dtype=torch.float32,
    attn_implementation="eager",
).eval()
config = model.config
decoder = Decoder(model, capacity).eval()
# The token count is symbolic; 8 only gives the trace an example.
tokens = torch.zeros((1, 8), dtype=torch.int64)
torch._dynamo.mark_dynamic(tokens, 1, min=1, max=capacity)
start = torch.zeros((1,), dtype=torch.int64)
shape = (1, config.num_key_value_heads, capacity, config.head_dim)
keys = [torch.zeros(shape) for _ in range(config.num_hidden_layers)]
values = [torch.zeros(shape) for _ in range(config.num_hidden_layers)]
compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=decompositions,
    trace=None,
)
with torch.no_grad():
    (graph,) = compiler.importer(decoder, tokens, start, keys, values)
params = compiler.imported_params[graph]
# Graph parameters are the checkpoint's parameters and buffers (inv_freq).
names = {
    value.data_ptr(): name
    for name, value in [*model.named_parameters(), *model.named_buffers()]
}
parameter_names = {
    node.name: names[value.data_ptr()]
    for node, value in zip(graph.params, params, strict=True)
}

# ===----------------------------------------------------------------------===
# Step 2.1: Export the original graph and FP32 parameters.
# ===----------------------------------------------------------------------===
graph.dump(output / "01-import.txt")
# graph.dump_parameters(output / "01-import-parameters")

# ===----------------------------------------------------------------------===
# Step 2.2: Absorb the transposes PyTorch puts on the Linear weights.
# ===----------------------------------------------------------------------===
permute(graph)
graph.dump(output / "02-permuted.txt")
# graph.dump_parameters(output / "02-permuted-parameters")

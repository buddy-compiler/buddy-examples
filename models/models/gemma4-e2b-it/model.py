# ===- model.py ----------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# Gemma-4-E2B-it model definition: one decoder step with its KV cache.
#
# ===---------------------------------------------------------------------------

from pathlib import Path

import torch
from transformers import AutoConfig, Gemma4ForCausalLM

checkpoint = Path(__file__).resolve().parent / "output/gemma4-e2b-it"


# The KV cache holds 256 positions.
capacity = 256


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
    """Tokens written from `start` into the KV cache: returns the logits of
    every token and the updated cache."""

    def __init__(self, model, capacity):
        super().__init__()
        self.model = model
        self.capacity = capacity

    def forward(self, tokens, start, keys, values):
        positions = start + torch.arange(tokens.shape[1])
        visible = torch.arange(keys[0].shape[2])[None, :] <= positions[:, None]
        sliding = visible & (
            torch.arange(keys[0].shape[2])[None, :]
            > positions[:, None] - self.model.config.sliding_window
        )
        mask = torch.where(visible, 0.0, torch.finfo(torch.float32).min)
        sliding_mask = torch.where(sliding, 0.0, torch.finfo(torch.float32).min)
        cache = Cache(start, keys, values)
        logits = self.model(
            input_ids=tokens,
            position_ids=positions[None, :],
            attention_mask={
                "full_attention": mask[None, None, :, :],
                "sliding_attention": sliding_mask[None, None, :, :],
            },
            past_key_values=cache,
            use_cache=True,
        ).logits
        return logits, cache.keys, cache.values


def load():
    """The decoder over the checkpoint, ready to import."""
    model = Gemma4ForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    return Decoder(model, capacity).eval()


def inputs(length):
    """Example inputs of the graph: `length` tokens written from position 0
    into an empty KV cache."""
    config = AutoConfig.from_pretrained(checkpoint)
    # The last `num_kv_shared_layers` layers reuse the KV states of earlier
    # layers inside the HF model: only the layers before them have cache
    # entries. Full-attention layers use the global head size.
    layers = config.layer_types[
        : config.num_hidden_layers - config.num_kv_shared_layers
    ]
    shapes = [
        (
            1,
            config.num_key_value_heads,
            capacity,
            config.global_head_dim if layer == "full_attention" else config.head_dim,
        )
        for layer in layers
    ]
    return (
        torch.zeros((1, length), dtype=torch.int64),
        torch.zeros((1,), dtype=torch.int64),
        [torch.zeros(shape) for shape in shapes],
        [torch.zeros(shape) for shape in shapes],
    )

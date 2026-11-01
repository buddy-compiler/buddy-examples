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
# Qwen3-0.6B model definition: one decoder step with its KV cache.
#
# ===---------------------------------------------------------------------------

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

checkpoint = Path(__file__).resolve().parent / "output/qwen3-0-6b"


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


def load():
    """The decoder over the checkpoint, ready to import."""
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    return Decoder(model, capacity).eval()


def inputs(length):
    """Example inputs of the graph: `length` tokens written from position 0
    into an empty KV cache."""
    config = AutoConfig.from_pretrained(checkpoint)
    shape = (1, config.num_key_value_heads, capacity, config.head_dim)
    return (
        torch.zeros((1, length), dtype=torch.int64),
        torch.zeros((1,), dtype=torch.int64),
        [torch.zeros(shape) for _ in range(config.num_hidden_layers)],
        [torch.zeros(shape) for _ in range(config.num_hidden_layers)],
    )

import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, weights, config, parts):
        super().__init__()
        self.weights = nn.ParameterDict(
            {name: nn.Parameter(value) for name, value in weights.items()}
        )
        self.head_dim = config["head_dim"]
        self.heads = config["num_attention_heads"] // parts
        self.kv_heads = config["num_key_value_heads"] // parts
        self.epsilon = config["rms_norm_eps"]
        rope = config["rope_scaling"]
        if rope["rope_type"] != "default" or not rope["mrope_interleaved"]:
            raise ValueError("Qwen3-Omni attention requires default interleaved MRoPE")
        frequencies = 1.0 / (
            config["rope_theta"]
            ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        indices = torch.arange(self.head_dim // 2)
        self.register_buffer("frequencies", frequencies)
        self.register_buffer(
            "horizontal",
            ((indices % 3 == 1) & (indices < rope["mrope_section"][1] * 3)).float(),
        )
        self.register_buffer(
            "vertical",
            ((indices % 3 == 2) & (indices < rope["mrope_section"][2] * 3)).float(),
        )

    def forward(self, hidden, keys, values, cache_positions, positions):
        length = hidden.shape[1]
        normalized = hidden * torch.rsqrt(
            hidden.square().mean(-1, keepdim=True) + self.epsilon
        )
        normalized = normalized * self.weights["norm"]
        query = F.linear(normalized, self.weights["q"]).reshape(
            1, length, self.heads, self.head_dim
        )
        key = F.linear(normalized, self.weights["k"]).reshape(
            1, length, self.kv_heads, self.head_dim
        )
        value = F.linear(normalized, self.weights["v"]).reshape(
            1, length, self.kv_heads, self.head_dim
        )
        query = query * torch.rsqrt(
            query.square().mean(-1, keepdim=True) + self.epsilon
        )
        query = (query * self.weights["q_norm"]).transpose(1, 2)
        key = key * torch.rsqrt(key.square().mean(-1, keepdim=True) + self.epsilon)
        key = (key * self.weights["k_norm"]).transpose(1, 2)
        value = value.transpose(1, 2)
        positions = positions.float()
        angles = (
            positions[0, :, None] * (1 - self.horizontal - self.vertical)
            + positions[1, :, None] * self.horizontal
            + positions[2, :, None] * self.vertical
        )
        angles = angles * self.frequencies
        angles = torch.cat((angles, angles), dim=-1)[None, None]
        cosine, sine = angles.cos(), angles.sin()
        half = self.head_dim // 2
        query = (
            query * cosine
            + torch.cat((-query[..., half:], query[..., :half]), dim=-1) * sine
        )
        key = (
            key * cosine + torch.cat((-key[..., half:], key[..., :half]), dim=-1) * sine
        )
        keys = keys.index_copy(2, cache_positions, key)
        values = values.index_copy(2, cache_positions, value)
        repeats = self.heads // self.kv_heads
        capacity = keys.shape[2]
        grouped_query = query.reshape(1, self.kv_heads, repeats * length, self.head_dim)
        scores = (grouped_query @ keys.transpose(-2, -1)) * self.head_dim**-0.5
        scores = scores.reshape(1, self.heads, length, capacity)
        mask = torch.arange(capacity)[None] > cache_positions[:, None]
        scores = scores.masked_fill(mask[None, None], torch.finfo(torch.float32).min)
        probabilities = scores.softmax(-1).reshape(
            1, self.kv_heads, repeats * length, capacity
        )
        output = (probabilities @ values).reshape(1, self.heads, length, self.head_dim)
        output = output.transpose(1, 2).reshape(1, length, -1)
        return F.linear(output, self.weights["o"]), keys, values

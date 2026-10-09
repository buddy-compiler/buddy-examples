import copy

import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input_ids):
        return self.embedding(input_ids)


def shard_linear(linear, axis, rank, parts):
    result = copy.deepcopy(linear)
    width = linear.weight.shape[axis] // parts
    result.weight = nn.Parameter(linear.weight.narrow(axis, rank * width, width).detach().clone())
    result.out_features, result.in_features = result.weight.shape
    return result


class Attention(nn.Module):
    def __init__(self, layer, config, cache_length, frequencies, rank, parts, attention_scaling):
        super().__init__()
        self.norm = layer.input_layernorm
        self.q_norm = layer.self_attn.q_norm
        self.k_norm = layer.self_attn.k_norm
        self.head_dim = config.head_dim
        self.scaling = layer.self_attn.scaling
        self.q_proj = shard_linear(layer.self_attn.q_proj, 0, rank, parts)
        self.k_proj = shard_linear(layer.self_attn.k_proj, 0, rank, parts)
        self.v_proj = shard_linear(layer.self_attn.v_proj, 0, rank, parts)
        self.o_proj = shard_linear(layer.self_attn.o_proj, 1, rank, parts)
        self.heads = config.num_attention_heads // parts
        self.kv_heads = config.num_key_value_heads // parts
        self.cache_length = cache_length
        self.register_buffer("frequencies", frequencies)
        self.attention_scaling = attention_scaling

    def forward(self, hidden, keys, values, positions):
        length = hidden.shape[1]
        normalized = self.norm(hidden)
        query = self.q_norm(
            self.q_proj(normalized).reshape(1, length, self.heads, self.head_dim)
        ).transpose(1, 2)
        key = self.k_norm(
            self.k_proj(normalized).reshape(1, length, self.kv_heads, self.head_dim)
        ).transpose(1, 2)
        value = self.v_proj(normalized).reshape(
            1, length, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        angles = positions.to(torch.float32)[:, None] * self.frequencies[None, :]
        angles = torch.cat((angles, angles), dim=-1)[None, None, :, :]
        cosine = angles.cos() * self.attention_scaling
        sine = angles.sin() * self.attention_scaling
        half = self.head_dim // 2
        query = query * cosine + torch.cat((-query[..., half:], query[..., :half]), dim=-1) * sine
        key = key * cosine + torch.cat((-key[..., half:], key[..., :half]), dim=-1) * sine
        keys = keys.index_copy(2, positions, key)
        values = values.index_copy(2, positions, value)
        repeats = self.heads // self.kv_heads
        all_keys = keys[:, :, None, :, :].expand(
            1, self.kv_heads, repeats, self.cache_length, self.head_dim
        ).reshape(1, self.heads, self.cache_length, self.head_dim)
        all_values = values[:, :, None, :, :].expand(
            1, self.kv_heads, repeats, self.cache_length, self.head_dim
        ).reshape(1, self.heads, self.cache_length, self.head_dim)
        scores = (query @ all_keys.transpose(-2, -1)) * self.scaling
        mask = torch.arange(self.cache_length)[None, :] > positions[:, None]
        scores = scores.masked_fill(mask[None, None, :, :], torch.finfo(torch.float32).min)
        output = (scores.softmax(-1) @ all_values).transpose(1, 2).reshape(
            1, length, self.heads * self.head_dim
        )
        return self.o_proj(output), keys, values


class FFN(nn.Module):
    def __init__(self, layer, rank, parts):
        super().__init__()
        self.norm = layer.post_attention_layernorm
        self.act_fn = layer.mlp.act_fn
        self.gate_proj = shard_linear(layer.mlp.gate_proj, 0, rank, parts)
        self.up_proj = shard_linear(layer.mlp.up_proj, 0, rank, parts)
        self.down_proj = shard_linear(layer.mlp.down_proj, 1, rank, parts)

    def forward(self, hidden):
        normalized = self.norm(hidden)
        return self.down_proj(self.act_fn(self.gate_proj(normalized)) * self.up_proj(normalized))

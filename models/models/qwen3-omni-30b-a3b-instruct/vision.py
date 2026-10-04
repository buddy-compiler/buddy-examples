import torch
from torch import nn
from torch.nn import functional as F


class Patch(nn.Module):
    def __init__(self, projection, bias, positions):
        super().__init__()
        self.projection = nn.Parameter(projection.flatten(1))
        self.bias = nn.Parameter(bias)
        self.positions = nn.Parameter(positions)

    def forward(self, patches, indices, coefficients):
        hidden = F.linear(patches, self.projection) + self.bias
        positions = F.embedding(indices, self.positions)
        return hidden + (positions * coefficients[:, :, None]).sum(1)


class Attention(nn.Module):
    def __init__(self, tensors, config):
        super().__init__()
        self.weights = nn.ParameterDict(
            {k: nn.Parameter(v) for k, v in tensors.items()}
        )
        self.heads = config["num_heads"]
        self.head_dim = config["hidden_size"] // self.heads

    def forward(self, hidden, cosine, sine, mask):
        length, width = hidden.shape
        normalized = F.layer_norm(
            hidden, (width,), self.weights["norm"], self.weights["norm_bias"], 1e-6
        )
        qkv = F.linear(normalized, self.weights["qkv"]) + self.weights["qkv_bias"]
        qkv = qkv.reshape(length, 3, self.heads, self.head_dim).permute(1, 2, 0, 3)
        query, key, value = qkv[0], qkv[1], qkv[2]
        half = self.head_dim // 2
        query = (
            query * cosine[None]
            + torch.cat((-query[..., half:], query[..., :half]), -1) * sine[None]
        )
        key = (
            key * cosine[None]
            + torch.cat((-key[..., half:], key[..., :half]), -1) * sine[None]
        )
        scores = (query @ key.transpose(-2, -1)) * self.head_dim**-0.5
        scores = scores.masked_fill(mask[None], torch.finfo(torch.float32).min)
        attended = (scores.softmax(-1) @ value).transpose(0, 1).reshape(length, width)
        return (
            hidden
            + F.linear(attended, self.weights["projection"])
            + self.weights["projection_bias"]
        )


class MLP(nn.Module):
    def __init__(self, tensors):
        super().__init__()
        self.weights = nn.ParameterDict(
            {k: nn.Parameter(v) for k, v in tensors.items()}
        )

    def forward(self, hidden):
        normalized = F.layer_norm(
            hidden,
            (hidden.shape[-1],),
            self.weights["norm"],
            self.weights["norm_bias"],
            1e-6,
        )
        inner = F.linear(normalized, self.weights["up"]) + self.weights["up_bias"]
        return (
            hidden
            + F.linear(F.gelu(inner, approximate="tanh"), self.weights["down"])
            + self.weights["down_bias"]
        )


class Merger(nn.Module):
    def __init__(self, tensors, config, postshuffle):
        super().__init__()
        self.weights = nn.ParameterDict(
            {k: nn.Parameter(v) for k, v in tensors.items()}
        )
        self.width = config["hidden_size"] * config["spatial_merge_size"] ** 2
        self.postshuffle = postshuffle

    def forward(self, hidden):
        if self.postshuffle:
            hidden = hidden.reshape(-1, self.width)
        normalized = F.layer_norm(
            hidden,
            (hidden.shape[-1],),
            self.weights["norm"],
            self.weights["norm_bias"],
            1e-6,
        )
        normalized = normalized.reshape(-1, self.width)
        inner = F.linear(normalized, self.weights["up"]) + self.weights["up_bias"]
        return F.linear(F.gelu(inner), self.weights["down"]) + self.weights["down_bias"]

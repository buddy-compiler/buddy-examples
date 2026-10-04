import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, tensors, config):
        super().__init__()
        self.weights = nn.ParameterDict(
            {k: nn.Parameter(v) for k, v in tensors.items()}
        )
        self.heads = config["num_attention_heads"]
        self.head_dim = config["hidden_size"] // self.heads
        self.epsilon = config["rms_norm_eps"]
        self.window = config["sliding_window"]
        frequency = 1.0 / (
            config["rope_theta"]
            ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("frequency", frequency)

    def forward(self, hidden, positions):
        length, width = hidden.shape
        normalized = (
            hidden
            * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + self.epsilon)
            * self.weights["norm"]
        )
        query = (
            F.linear(normalized, self.weights["q"])
            .reshape(length, self.heads, self.head_dim)
            .transpose(0, 1)
        )
        key = (
            F.linear(normalized, self.weights["k"])
            .reshape(length, self.heads, self.head_dim)
            .transpose(0, 1)
        )
        value = (
            F.linear(normalized, self.weights["v"])
            .reshape(length, self.heads, self.head_dim)
            .transpose(0, 1)
        )
        angles = positions[:, None].float() * self.frequency
        angles = torch.cat((angles, angles), -1)[None]
        cosine, sine = angles.cos(), angles.sin()
        half = self.head_dim // 2
        query = (
            query * cosine
            + torch.cat((-query[..., half:], query[..., :half]), -1) * sine
        )
        key = key * cosine + torch.cat((-key[..., half:], key[..., :half]), -1) * sine
        scores = (query @ key.transpose(-2, -1)) * self.head_dim**-0.5
        mask = (positions[None] > positions[:, None]) | (
            positions[None] <= positions[:, None] - self.window
        )
        scores = scores.masked_fill(mask[None], torch.finfo(torch.float32).min)
        result = (scores.softmax(-1) @ value).transpose(0, 1).reshape(length, width)
        return hidden + F.linear(result, self.weights["o"]) * self.weights["scale"]


class Dense(nn.Module):
    def __init__(self, tensors, epsilon):
        super().__init__()
        self.weights = nn.ParameterDict(
            {k: nn.Parameter(v) for k, v in tensors.items()}
        )
        self.epsilon = epsilon

    def forward(self, hidden):
        normalized = (
            hidden
            * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + self.epsilon)
            * self.weights["norm"]
        )
        gate = F.silu(F.linear(normalized, self.weights["gate"]))
        up = F.linear(normalized, self.weights["up"])
        return (
            hidden + F.linear(gate * up, self.weights["down"]) * self.weights["scale"]
        )

import torch
from torch import nn
from torch.nn import functional as F


class Convolution(nn.Module):
    def __init__(self, weight, bias, channels):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.channels = channels

    def forward(self, features, frames):
        patches = F.pad(features, (1, 1, 1, 1)).unfold(2, 3, 2).unfold(3, 3, 2)
        height, length = patches.shape[2:4]
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, self.channels * 9)
        patches = F.pad(patches, (0, self.weight.shape[1] - patches.shape[1]))
        hidden = F.gelu(F.linear(patches, self.weight) + self.bias)
        hidden = hidden.reshape(1, height, length, -1).permute(0, 3, 1, 2)
        valid = torch.arange(length) < (frames + 1) // 2
        return hidden * valid[None, None, None]


class Projection(nn.Module):
    def __init__(self, weight, position_embedding):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.register_buffer("positions", position_embedding)

    def forward(self, features):
        batch, channels, height, length = features.shape
        features = features.permute(0, 3, 1, 2).reshape(
            batch * length, channels * height
        )
        return F.linear(features, self.weight) + self.positions[:length]


class Attention(nn.Module):
    def __init__(self, tensors, config):
        super().__init__()
        self.weights = nn.ParameterDict(
            {k: nn.Parameter(v) for k, v in tensors.items()}
        )
        self.heads = config["encoder_attention_heads"]
        self.head_dim = config["d_model"] // self.heads

    def forward(self, hidden, mask):
        length, width = hidden.shape
        normalized = F.layer_norm(
            hidden, (width,), self.weights["norm"], self.weights["norm_bias"], 1e-5
        )
        qkv = F.linear(normalized, self.weights["qkv"]) + self.weights["qkv_bias"]
        qkv = qkv.reshape(length, 3, self.heads, self.head_dim).permute(1, 2, 0, 3)
        query, key, value = qkv[0], qkv[1], qkv[2]
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
            1e-5,
        )
        inner = F.linear(normalized, self.weights["up"]) + self.weights["up_bias"]
        return (
            hidden
            + F.linear(F.gelu(inner), self.weights["down"])
            + self.weights["down_bias"]
        )


class Output(nn.Module):
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
            1e-5,
        )
        inner = F.linear(normalized, self.weights["up"]) + self.weights["up_bias"]
        return F.linear(F.gelu(inner), self.weights["down"]) + self.weights["down_bias"]

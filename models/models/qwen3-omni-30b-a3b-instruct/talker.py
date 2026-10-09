import torch
from torch import nn
from torch.nn import functional as F


class Router(nn.Module):
    def __init__(self, norm, gate, shared_gate, config):
        super().__init__()
        self.norm = nn.Parameter(norm)
        self.gate = nn.Parameter(gate)
        self.shared_gate = nn.Parameter(shared_gate)
        self.epsilon = config["rms_norm_eps"]
        self.top_k = config["num_experts_per_tok"]

    def forward(self, hidden):
        normalized = hidden * torch.rsqrt(
            hidden.square().mean(-1, keepdim=True) + self.epsilon
        )
        normalized = normalized * self.norm
        scores, experts = (
            F.linear(normalized, self.gate).softmax(-1).topk(self.top_k, dim=-1)
        )
        scores = scores / scores.sum(-1, keepdim=True)
        shared = F.linear(normalized, self.shared_gate).sigmoid()
        return normalized, experts, scores, shared


class Resize(nn.Module):
    def __init__(self, up, up_bias, down, down_bias):
        super().__init__()
        self.up = nn.Parameter(up)
        self.up_bias = nn.Parameter(up_bias)
        self.down = nn.Parameter(down)
        self.down_bias = nn.Parameter(down_bias)

    def forward(self, hidden):
        value = F.silu(F.linear(hidden, self.up) + self.up_bias)
        return F.linear(value, self.down) + self.down_bias


class Dense(nn.Module):
    def __init__(self, norm, gate, up, down, epsilon):
        super().__init__()
        self.norm = nn.Parameter(norm)
        self.gate = nn.Parameter(gate)
        self.up = nn.Parameter(up)
        self.down = nn.Parameter(down)
        self.epsilon = epsilon

    def forward(self, hidden):
        normalized = hidden * torch.rsqrt(
            hidden.square().mean(-1, keepdim=True) + self.epsilon
        )
        normalized = normalized * self.norm
        return F.linear(
            F.silu(F.linear(normalized, self.gate)) * F.linear(normalized, self.up),
            self.down,
        )

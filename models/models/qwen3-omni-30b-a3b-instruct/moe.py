import torch
from torch import nn
from torch.nn import functional as F


class Router(nn.Module):
    def __init__(self, norm, gate, *, epsilon, top_k, normalize):
        super().__init__()
        self.norm = nn.Parameter(norm)
        self.gate = nn.Parameter(gate)
        self.epsilon = epsilon
        self.top_k = top_k
        self.normalize = normalize

    def forward(self, hidden):
        hidden = hidden.float()
        normalized = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + self.epsilon)
        normalized = normalized * self.norm
        probabilities = F.linear(normalized, self.gate).softmax(-1)
        scores, experts = probabilities.topk(self.top_k, dim=-1)
        if self.normalize:
            scores = scores / scores.sum(-1, keepdim=True)
        return normalized, experts, scores


class Expert(nn.Module):
    def __init__(self, gate, up, down):
        super().__init__()
        self.gate = nn.Parameter(gate)
        self.up = nn.Parameter(up)
        self.down = nn.Parameter(down)

    def forward(self, hidden):
        return F.linear(F.silu(F.linear(hidden, self.gate)) * F.linear(hidden, self.up), self.down)

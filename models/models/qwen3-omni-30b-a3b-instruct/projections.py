import torch
from torch import nn
from torch.nn import functional as F


class Embedding(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, tokens, begin):
        indices = tokens - begin
        owned = (indices >= 0) & (indices < self.weight.shape[0])
        indices = torch.where(owned, indices, 0)
        return F.embedding(indices, self.weight) * owned.unsqueeze(-1)


class Norm(nn.Module):
    def __init__(self, weight, epsilon):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.epsilon = epsilon

    def forward(self, hidden):
        hidden = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + self.epsilon)
        return hidden * self.weight


class Output(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, hidden):
        return F.linear(hidden, self.weight)

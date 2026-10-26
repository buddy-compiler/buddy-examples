import torch
from torch import nn


class Mlp(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, value):
        return self.fc2(self.act(self.fc1(value)))


class MixerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(768, eps=1e-6)
        self.mlp_tokens = Mlp(196, 384)
        self.norm2 = nn.LayerNorm(768, eps=1e-6)
        self.mlp_channels = Mlp(768, 3072)

    def forward(self, value):
        mixed = self.mlp_tokens(self.norm1(value).transpose(0, 1))
        value = value + mixed.transpose(0, 1)
        return value + self.mlp_channels(self.norm2(value))


class Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Module()
        self.stem.proj = nn.Linear(768, 768)
        self.blocks = nn.Sequential(*(MixerBlock() for _ in range(12)))
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.head = nn.Linear(768, 1000)

    def forward(self, patches):
        value = self.blocks(self.stem.proj(patches))
        return self.head(self.norm(value).mean(dim=0, keepdim=True))

from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

checkpoint = Path(__file__).resolve().parent / "output/mlp-mixer"


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
        mixed = self.mlp_tokens(self.norm1(value).transpose(-2, -1))
        value = value + mixed.transpose(-2, -1)
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
        """[196, 768] patches to [1, 1000] logits; a leading batch axis, if
        any, carries through."""
        value = self.blocks(self.stem.proj(patches))
        # The patch axis as a non-negative dim: the traced graph stays as for
        # [196, 768].
        return self.head(self.norm(value).mean(dim=value.ndim - 2, keepdim=True))


def load():
    """Mixer-B/16 with the checkpoint weights, ready to import."""
    state = load_file(checkpoint / "model.safetensors")
    # The non-overlapping 16x16 patch projection is a linear map: its [C, H, W]
    # weight dimensions flatten to the [196, 768] patch input.
    state["stem.proj.weight"] = state["stem.proj.weight"].flatten(1)
    model = Mixer().eval()
    model.load_state_dict(state, strict=True)
    return model


def inputs():
    """Example inputs of the graph: 196 patches of [C, 16, 16]."""
    return (torch.zeros((196, 768)),)

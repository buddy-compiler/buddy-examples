from pathlib import Path
import tomllib

from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch
from torch import nn


with Path(__file__).with_name("configs").joinpath("model.toml").open("rb") as file:
    PARAMETERS = tomllib.load(file)["parameters"]


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


def load_model(checkpoint=PARAMETERS["checkpoint"], revision=PARAMETERS["revision"]):
    path = hf_hub_download(checkpoint, "model.safetensors", revision=revision)
    state = load_file(path)
    # The non-overlapping 16x16 patch projection is a linear map. Flatten
    # its C,H,W weight dimensions to match the prepared patch input layout.
    state["stem.proj.weight"] = state["stem.proj.weight"].flatten(1)
    model = Mixer().eval()
    model.load_state_dict(state, strict=True)
    return model


def load_pixels(path: Path):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    size = 256
    target = (size, int(size * height / width)) if width <= height else (int(size * width / height), size)
    image = image.resize(target, Image.Resampling.BICUBIC)
    left, top = round((image.width - 224) / 2), round((image.height - 224) / 2)
    image = image.crop((left, top, left + 224, top + 224))
    pixels = np.asarray(image, dtype=np.float32) / np.float32(127.5) - np.float32(1.0)
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy())[None]


def pack_patches(pixels):
    # Offline input layout: patch-row, patch-column, channel, local-row,
    # local-column. No high-rank transpose remains in the compiled graph.
    return pixels.reshape(1, 3, 14, 16, 14, 16).permute(0, 2, 4, 1, 3, 5).reshape(196, 768).contiguous()


def load_input(path: Path):
    return pack_patches(load_pixels(path))

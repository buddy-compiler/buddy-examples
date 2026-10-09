import importlib
import math

import torch
from torch.nn import functional as F

Checkpoint = importlib.import_module(
    "stack.models.models.qwen3-omni-30b-a3b-instruct.weights"
).Checkpoint

stages = importlib.import_module(
    "stack.models.models.qwen3-omni-30b-a3b-instruct.audio"
)


class Weights(Checkpoint):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.config = self.metadata["thinker_config"]["audio_config"]

    def convolution(self, index):
        prefix = f"thinker.audio_tower.conv2d{index}"
        weight = self.tensor(f"{prefix}.weight")
        channels = weight.shape[1]
        weight = weight.flatten(1)
        weight = F.pad(weight, (0, (-weight.shape[1]) % 32))
        return stages.Convolution(
            weight, self.tensor(f"{prefix}.bias"), channels
        ).eval()

    def projection(self):
        config = self.config
        half = config["d_model"] // 2
        frequencies = torch.exp(
            -math.log(10000.0) / (half - 1) * torch.arange(half).float()
        )
        angles = torch.arange(config["max_source_positions"])[:, None] * frequencies
        positions = torch.cat((angles.sin(), angles.cos()), -1)
        return stages.Projection(
            self.tensor("thinker.audio_tower.conv_out.weight"), positions
        ).eval()

    def attention(self, layer):
        prefix = f"thinker.audio_tower.layers.{layer}"
        tensors = {
            "norm": self.tensor(f"{prefix}.self_attn_layer_norm.weight"),
            "norm_bias": self.tensor(f"{prefix}.self_attn_layer_norm.bias"),
            "projection": self.tensor(f"{prefix}.self_attn.out_proj.weight"),
            "projection_bias": self.tensor(f"{prefix}.self_attn.out_proj.bias"),
        }
        tensors["qkv"] = torch.cat(
            [
                self.tensor(f"{prefix}.self_attn.{name}_proj.weight")
                for name in ("q", "k", "v")
            ]
        )
        tensors["qkv_bias"] = torch.cat(
            [
                self.tensor(f"{prefix}.self_attn.{name}_proj.bias")
                for name in ("q", "k", "v")
            ]
        )
        return stages.Attention(tensors, self.config).eval()

    def mlp(self, layer):
        prefix = f"thinker.audio_tower.layers.{layer}"
        names = {
            "norm": "final_layer_norm.weight",
            "norm_bias": "final_layer_norm.bias",
            "up": "fc1.weight",
            "up_bias": "fc1.bias",
            "down": "fc2.weight",
            "down_bias": "fc2.bias",
        }
        return stages.MLP(
            {k: self.tensor(f"{prefix}.{v}") for k, v in names.items()}
        ).eval()

    def output(self):
        prefix = "thinker.audio_tower"
        names = {
            "norm": "ln_post.weight",
            "norm_bias": "ln_post.bias",
            "up": "proj1.weight",
            "up_bias": "proj1.bias",
            "down": "proj2.weight",
            "down_bias": "proj2.bias",
        }
        return stages.Output(
            {k: self.tensor(f"{prefix}.{v}") for k, v in names.items()}
        ).eval()

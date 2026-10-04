import copy

import torch
from torch import nn

def ffn_widths(size, alignment):
    if size % alignment:
        raise ValueError("FFN dimensions must follow the packed weight alignment")
    width = ((size // alignment + 2) // 3) * alignment
    last = size - 2 * width
    if last <= 0:
        raise ValueError("FFN dimension must accommodate three workers")
    return width, width, last


class FFNExpand(nn.Module):
    def __init__(self, stage, width):
        super().__init__()
        self.norm = stage.norm
        self.act_fn = stage.act_fn
        self.gate_proj = copy.deepcopy(stage.gate_proj)
        self.up_proj = copy.deepcopy(stage.up_proj)
        for projection in (self.gate_proj, self.up_proj):
            projection.weight = nn.Parameter(projection.weight[:width].detach().clone())
            projection.out_features = width

    def forward(self, hidden):
        normalized = self.norm(hidden)
        return self.act_fn(self.gate_proj(normalized)) * self.up_proj(normalized)


class FFNDown(nn.Module):
    def __init__(self, stage, width):
        super().__init__()
        self.down_proj = copy.deepcopy(stage.down_proj)
        self.down_proj.weight = nn.Parameter(self.down_proj.weight[:width].detach().clone())
        self.down_proj.out_features = width

    def forward(self, hidden):
        return self.down_proj(hidden)


def parameters(layer, rank, parts, expand_widths, down_widths):
    intermediate = layer.mlp.gate_proj.weight.shape[0] // parts
    hidden = layer.mlp.down_proj.weight.shape[0]
    group, half = rank % parts, rank // parts
    values = {"norm.weight": layer.post_attention_layernorm.weight}
    begin = group * intermediate + half * intermediate // 2
    for index, width in enumerate(expand_widths):
        values[f"gate_{index}.weight"] = layer.mlp.gate_proj.weight[begin:begin + width]
        values[f"up_{index}.weight"] = layer.mlp.up_proj.weight[begin:begin + width]
        begin += width
    begin = half * hidden // 2
    for index, width in enumerate(down_widths):
        values[f"down_{index}.weight"] = layer.mlp.down_proj.weight[
            begin:begin + width, group * intermediate:(group + 1) * intermediate]
        begin += width
    return values

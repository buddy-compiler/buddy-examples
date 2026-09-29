# Laya architecture: https://github.com/NandhaKishorM/laya (Apache-2.0).
import copy
import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoConfig, ModernBertModel

CHECKPOINT = "convaiinnovations/laya"
REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"


class Linear(nn.Linear):
    def forward(self, x):
        result = x @ self.weight.T
        if self.bias is not None:
            result = result + self.bias
        return result


def linear(source):
    result = Linear(source.in_features, source.out_features, source.bias is not None)
    result.load_state_dict(source.state_dict())
    return result


class Model(nn.Module):
    def __init__(self, directory):
        super().__init__()
        directory = Path(directory)
        self.settings = json.loads((directory / "rl_agent_config.json").read_text())
        config = AutoConfig.from_pretrained(directory / "encoder")
        config._attn_implementation = "eager"
        config.reference_compile = False
        self.encoder = ModernBertModel(config)
        d = config.hidden_size
        layer = nn.TransformerEncoderLayer(
            d, d // 64, 4 * d, 0.1, batch_first=True, norm_first=True
        )
        self.head = nn.TransformerEncoder(
            layer, self.settings["head_layers"], enable_nested_tensor=False
        )
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.act_head = nn.Sequential(
            nn.Linear(d + 4, 256),
            nn.GELU(),
            nn.Linear(256, len(self.settings["act_costs"]) + 1),
        )
        self.register_buffer("temperature", torch.ones(3))
        self.load_state_dict(load_file(directory / "model.safetensors"), strict=True)
        self.eval()

    def forward(self, tokens, mask, positions, valid, qtype):
        hidden = self.encoder(input_ids=tokens, attention_mask=mask).last_hidden_state
        hidden = self.head(
            hidden + self.type_emb(qtype)[:, None, :], src_key_padding_mask=~mask.bool()
        )
        markers = hidden.gather(
            1, positions[:, :, None].expand(-1, -1, hidden.shape[-1])
        )
        logits = self.scorer(markers).squeeze(-1).masked_fill(~valid, -1e4)
        probabilities = logits.softmax(-1)
        count = valid.sum(-1).clamp(min=2).float()
        entropy = (
            -(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1) / count.log()
        )
        top = probabilities.topk(2, -1).values
        features = torch.stack(
            (top[:, 0], top[:, 0] - top[:, 1], entropy, count / 255), -1
        )
        return logits, self.act_head(torch.cat((hidden[:, 0], features), -1))


class Attention(nn.Module):
    def __init__(self, layer, config, length):
        super().__init__()
        self.norm = layer.attn_norm
        self.qkv = linear(layer.attn.Wqkv)
        self.out = linear(layer.attn.Wo)
        self.heads = config.num_attention_heads
        self.dim = config.hidden_size // self.heads
        theta = config.rope_parameters[layer.attention_type]["rope_theta"]
        angles = torch.arange(length)[:, None] / theta ** (
            torch.arange(0, self.dim, 2).float()[None, :] / self.dim
        )
        angles = torch.cat((angles, angles), -1)[None, None]
        self.register_buffer("cosine", angles.cos())
        self.register_buffer("sine", angles.sin())
        self.local = (
            config.local_attention // 2
            if layer.attention_type == "sliding_attention"
            else length
        )

    def forward(self, hidden, mask):
        length = hidden.shape[1]
        qkv = self.qkv(self.norm(hidden)).reshape(1, length, 3, self.heads, self.dim)
        q, k, v = (qkv[:, :, index].transpose(1, 2) for index in range(3))
        half = self.dim // 2
        q = q * self.cosine + torch.cat((-q[..., half:], q[..., :half]), -1) * self.sine
        k = k * self.cosine + torch.cat((-k[..., half:], k[..., :half]), -1) * self.sine
        scores = (q @ k.transpose(-2, -1)) * self.dim**-0.5
        indices = torch.arange(length)
        invalid = (indices[:, None] - indices[None, :]).abs() > self.local
        invalid = invalid[None, None] | ~mask[:, None, None, :].bool()
        scores = scores.masked_fill(invalid, torch.finfo(torch.float32).min)
        result = (scores.softmax(-1) @ v).transpose(1, 2).reshape(1, length, -1)
        return hidden + self.out(result)


class FFN(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.norm = layer.mlp_norm
        self.up = linear(layer.mlp.Wi)
        self.down = linear(layer.mlp.Wo)
        self.activation = layer.mlp.act

    def forward(self, hidden):
        value, gate = self.up(self.norm(hidden)).chunk(2, -1)
        return hidden + self.down(self.activation(value) * gate)


class HeadAttention(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.norm = layer.norm1
        d = layer.self_attn.embed_dim
        self.qkv = Linear(d, 3 * d)
        self.qkv.weight = nn.Parameter(layer.self_attn.in_proj_weight.detach().clone())
        self.qkv.bias = nn.Parameter(layer.self_attn.in_proj_bias.detach().clone())
        self.out = linear(layer.self_attn.out_proj)
        self.heads = layer.self_attn.num_heads
        self.dim = d // self.heads

    def forward(self, hidden, mask):
        length = hidden.shape[1]
        qkv = self.qkv(self.norm(hidden)).reshape(1, length, 3, self.heads, self.dim)
        q, k, v = (qkv[:, :, index].transpose(1, 2) for index in range(3))
        scores = (q @ k.transpose(-2, -1)) * self.dim**-0.5
        scores = scores.masked_fill(
            ~mask[:, None, None, :].bool(), torch.finfo(torch.float32).min
        )
        result = (scores.softmax(-1) @ v).transpose(1, 2).reshape(1, length, -1)
        return hidden + self.out(result)


class HeadFFN(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.norm = layer.norm2
        self.up = linear(layer.linear1)
        self.down = linear(layer.linear2)
        self.activation = layer.activation

    def forward(self, hidden):
        return hidden + self.down(self.activation(self.up(self.norm(hidden))))


class TypedHidden(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.norm = model.encoder.final_norm
        self.type_emb = model.type_emb

    def forward(self, hidden, qtype):
        return self.norm(hidden) + self.type_emb(qtype.reshape(1, 1))


class Projection(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.layers = copy.deepcopy(source)

    def forward(self, input):
        return self.layers(input)


def stages(model, length):
    yield "embedding", "embedding", model.encoder.embeddings
    for index, layer in enumerate(model.encoder.layers):
        yield f"attention_{index}", "attention", Attention(
            layer, model.encoder.config, length
        )
        yield f"ffn_{index}", "ffn", FFN(layer)
    yield "typed", "typed", TypedHidden(model)
    for index, layer in enumerate(model.head.layers):
        yield f"head_attention_{index}", "attention", HeadAttention(layer)
        yield f"head_ffn_{index}", "ffn", HeadFFN(layer)
    yield "scorer", "scorer", Projection(model.scorer)
    yield "action", "action", Projection(model.act_head)

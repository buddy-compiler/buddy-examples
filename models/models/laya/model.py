# Laya architecture: https://github.com/NandhaKishorM/laya (Apache-2.0).
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, ModernBertModel

checkpoint = Path(__file__).resolve().parent / "output/laya"


class Encoder(nn.Module):
    """Expose ModernBERT masks as tensor operations at the import boundary."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.shape[1])
        visible = attention_mask[:, None, None, :].bool()
        local = (positions[:, None] - positions[None, :]).abs()
        sliding = visible & (local <= self.model.config.sliding_window)
        mask = torch.where(visible, 0.0, torch.finfo(torch.float32).min)
        sliding_mask = torch.where(sliding, 0.0, torch.finfo(torch.float32).min)
        return self.model(
            input_ids=input_ids,
            attention_mask={"full_attention": mask, "sliding_attention": sliding_mask},
        )


class Model(nn.Module):
    def __init__(self, settings, config):
        super().__init__()
        self.settings = settings
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

    def forward(self, tokens, mask, positions, valid, qtype):
        hidden = self.encoder(input_ids=tokens, attention_mask=mask).last_hidden_state
        hidden = self.head(
            hidden + self.type_emb(qtype)[:, None, :], src_key_padding_mask=~mask.bool()
        )
        markers = hidden.gather(
            1, positions[:, :, None].expand(-1, -1, hidden.shape[-1])
        )
        logits = self.scorer(markers).squeeze(-1).masked_fill(~valid.bool(), -1e4)
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


def load():
    """Laya with the checkpoint weights, ready to import."""
    settings = json.loads((checkpoint / "rl_agent_config.json").read_text())
    model = Model(settings, AutoConfig.from_pretrained(checkpoint / "encoder"))
    model.load_state_dict(load_file(checkpoint / "model.safetensors"), strict=True)
    model.eval()
    # The masks ModernBERT builds itself become the same tensor operations.
    model.encoder = Encoder(model.encoder).eval()
    # The transformer layers of the head run their standard operations, not
    # the fused inference kernel PyTorch substitutes in eval mode.
    torch.backends.mha.set_fastpath_enabled(False)
    return model


def inputs():
    """Example inputs of the graph: 128 tokens with 16 marker positions."""
    return (
        torch.zeros((1, 128), dtype=torch.int64),
        torch.ones((1, 128), dtype=torch.int64),
        torch.arange(16, dtype=torch.int64)[None, :],
        torch.ones((1, 16), dtype=torch.bool),
        torch.zeros((1,), dtype=torch.int64),
    )

# Laya architecture: https://github.com/NandhaKishorM/laya (Apache-2.0).
import torch
from torch import nn
from transformers import ModernBertModel


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

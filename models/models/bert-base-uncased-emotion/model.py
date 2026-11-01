from pathlib import Path

import torch
from transformers import BertForSequenceClassification

checkpoint = Path(__file__).resolve().parent / "output/bert-base-uncased-emotion"


class BertLogits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, token_type_ids, attention_mask, position_ids):
        hidden = self.model.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        mask = (1.0 - attention_mask[:, None, None, :].to(hidden.dtype)) * torch.finfo(
            hidden.dtype
        ).min
        for layer in self.model.bert.encoder.layer:
            hidden = layer(hidden, attention_mask=mask)
        pooled = self.model.bert.pooler(hidden)
        return self.model.classifier(self.model.dropout(pooled))


def load():
    """The emotion classifier with eager attention, ready to import."""
    return BertLogits(
        BertForSequenceClassification.from_pretrained(
            checkpoint, attn_implementation="eager"
        )
    ).eval()


def inputs():
    """Example inputs of the graph: one sequence of 32 tokens."""
    return (
        torch.zeros((1, 32), dtype=torch.int64),
        torch.zeros((1, 32), dtype=torch.int64),
        torch.ones((1, 32), dtype=torch.int64),
        torch.arange(32, dtype=torch.int64)[None, :],
    )

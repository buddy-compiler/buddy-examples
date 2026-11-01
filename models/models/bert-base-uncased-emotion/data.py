"""Emotion sentences as the graph takes them: tokenized with the checkpoint
vocabulary to 32 tokens, with their positions."""

from importlib import import_module
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import BertTokenizer

emotion = Path(__file__).resolve().parents[2] / "datasets/emotion/split"
bert = import_module("models.models.bert-base-uncased-emotion.model")
tokenizer = BertTokenizer.from_pretrained(bert.checkpoint)


def load(split, count):
    """The first `count` sentences of the split as the inputs of the graph,
    32 tokens each, and their labels."""
    rows = pq.read_table(emotion / f"{split}-00000-of-00001.parquet").slice(0, count)
    tokens = tokenizer(
        rows["text"].to_pylist(),
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    positions = torch.arange(32)[None, :].expand(count, 32)
    inputs = (
        tokens["input_ids"],
        tokens["token_type_ids"],
        tokens["attention_mask"],
        positions,
    )
    return inputs, torch.tensor(rows["label"].to_pylist())

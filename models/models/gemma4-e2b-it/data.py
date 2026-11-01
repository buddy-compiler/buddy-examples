"""WikiText-2 test text as the perplexity windows of the model: the rows
joined by blank lines, tokenized with the checkpoint vocabulary and cut into
windows of `capacity` tokens, each a BOS token and `capacity - 1` tokens of
the text: Gemma reads garbage without BOS."""

from importlib import import_module
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

model = import_module("models.models.gemma4-e2b-it.model")
wikitext = Path(__file__).resolve().parents[2] / "datasets/wikitext/wikitext-2-raw-v1"
tokenizer = AutoTokenizer.from_pretrained(model.checkpoint)


def windows(count):
    """The first `count` windows of the text."""
    text = "\n\n".join(
        pq.read_table(wikitext / "test-00000-of-00001.parquet")["text"].to_pylist()
    )
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    tokens = tokens[: count * (model.capacity - 1)].reshape(count, model.capacity - 1)
    return torch.cat([torch.full((count, 1), tokenizer.bos_token_id), tokens], 1)

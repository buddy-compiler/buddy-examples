#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import BertForSequenceClassification, BertTokenizer

MODEL_ID = "bhadresh-savani/bert-base-uncased-emotion"

SAMPLES = [
    ("I am feeling great today!", 1),
    ("This makes me so angry.", 0),
    ("I am scared about what happens next.", 4),
    ("What a wonderful surprise!", 1),
    ("I feel empty and hopeless.", 3),
]


def load_recon(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) % 4 != 0:
        raise ValueError(f"recon size {len(raw)} not multiple of 4")
    return np.frombuffer(raw, dtype=np.float32).copy()


def inject_weights(model: torch.nn.Module, recon: np.ndarray) -> None:
    dynamo = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        verbose=False,
    )
    tokenizer = BertTokenizer.from_pretrained(MODEL_ID)
    inputs = {
        "input_ids": torch.tensor([[101, 1045, 2572, 102]], dtype=torch.int64),
        "token_type_ids": torch.tensor([[0, 0, 0, 0]], dtype=torch.int64),
        "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.int64),
    }
    with torch.no_grad():
        graphs = dynamo.importer(model, **inputs)
    params = dynamo.imported_params[graphs[0]][:-1]
    cursor = 0
    with torch.no_grad():
        for param in params:
            n = param.numel()
            if cursor + n > recon.size:
                raise ValueError(f"recon truncated at {cursor}+{n}>{recon.size}")
            chunk = recon[cursor : cursor + n].reshape(param.shape)
            param.data.copy_(torch.from_numpy(chunk))
            cursor += n
    if cursor != recon.size:
        raise ValueError(f"recon leftover {recon.size - cursor} elements")


def eval_ppl(model: torch.nn.Module, tokenizer: BertTokenizer) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for text, label in SAMPLES:
            enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            logits = model(**enc).logits
            loss = F.cross_entropy(logits, torch.tensor([label]))
            losses.append(float(loss))
    return math.exp(sum(losses) / len(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"missing {args.weights}")

    recon = load_recon(args.weights)
    model = BertForSequenceClassification.from_pretrained(MODEL_ID)
    inject_weights(model, recon)
    tokenizer = BertTokenizer.from_pretrained(MODEL_ID)
    n = len(SAMPLES)
    ppl = eval_ppl(model, tokenizer)
    print(f"ppl_n{n}={ppl:.4f}  # small-sample relative metric only (n={n}, noisy)")


if __name__ == "__main__":
    main()

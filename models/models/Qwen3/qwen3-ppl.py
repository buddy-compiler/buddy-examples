#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoModelForCausalLM, StaticCache

EVAL_TEXT = (
    "The capital of France is Paris. Machine learning models can be quantized "
    "to reduce memory while preserving accuracy."
)
HERE = Path(__file__).resolve().parent
VOCAB = HERE / "vocab.txt"


def model_path() -> str:
    path = os.environ.get("QWEN3_0_6B_MODEL_PATH")
    if path:
        return path
    cache = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"
    if cache.is_dir():
        snaps = sorted(cache.iterdir())
        if snaps:
            return str(snaps[-1])
    return "Qwen/Qwen3-0.6B"


def load_vocab(path: Path) -> dict[str, int]:
    tok2id: dict[str, int] = {}
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            tok2id[line.rstrip("\n")] = i
    if not tok2id:
        raise ValueError(f"empty vocab: {path}")
    return tok2id


def tokenize(text: str, vocab: Path) -> list[int]:
    tok2id = load_vocab(vocab)
    id_len = [0] * len(tok2id)
    for tok, tid in tok2id.items():
        if tid >= len(id_len):
            raise ValueError(f"vocab id out of range: {tid}")
        id_len[tid] = len(tok)

    s = text.replace(" ", "Ġ") if " " in text else text
    n = len(s)
    score = [0.0] * (n + 1)
    prev = [0] * (n + 1)
    for i in range(n):
        for sub_len in range(1, n - i + 1):
            piece = s[i : i + sub_len]
            tid = tok2id.get(piece)
            if tid is None:
                continue
            local = score[i] + sub_len * sub_len
            nxt = i + sub_len
            if score[nxt] < local:
                score[nxt] = local
                prev[nxt] = tid
    if score[n] <= 0:
        raise ValueError("tokenization failed for eval text")

    ids: list[int] = []
    i = n
    while i > 0:
        tid = prev[i]
        if tid == 0 and i > 0:
            raise ValueError("tokenization backtrace failed")
        ids.append(tid)
        i -= id_len[tid]
    ids.reverse()
    return ids


def load_recon_bf16(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) % 2 != 0:
        raise ValueError(f"recon size {len(raw)} not multiple of 2")
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def inject_weights(model: torch.nn.Module, recon_f32: np.ndarray) -> None:
    dynamo = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        verbose=False,
    )
    with torch.no_grad():
        past = StaticCache(config=model.config, max_cache_len=16)
        graphs = dynamo.importer(
            model,
            input_ids=torch.zeros((1, 16), dtype=torch.int64),
            past_key_values=past,
            use_cache=True,
        )
    params = dynamo.imported_params[graphs[0]]
    cursor = 0
    with torch.no_grad():
        for param in params:
            if param.dtype != torch.bfloat16:
                continue
            n = param.numel()
            if cursor + n > recon_f32.size:
                raise ValueError(f"recon truncated at {cursor}+{n}>{recon_f32.size}")
            chunk = torch.from_numpy(recon_f32[cursor : cursor + n].reshape(param.shape))
            param.data.copy_(chunk.to(torch.bfloat16))
            cursor += n
    if cursor != recon_f32.size:
        raise ValueError(f"recon leftover {recon_f32.size - cursor} elements")


def eval_ppl(model: torch.nn.Module) -> float:
    model.eval()
    ids = tokenize(EVAL_TEXT, VOCAB)
    input_ids = torch.tensor([ids], dtype=torch.int64)
    if input_ids.shape[1] < 2:
        raise ValueError("eval text too short for PPL")
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
        loss = float(out.loss)
    return math.exp(loss)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"missing {args.weights}")

    recon = load_recon_bf16(args.weights)
    model = AutoModelForCausalLM.from_pretrained(
        model_path(), local_files_only=True, torch_dtype=torch.bfloat16
    )
    inject_weights(model, recon)
    ppl = eval_ppl(model)
    print(f"ppl={ppl:.4f}")


if __name__ == "__main__":
    main()

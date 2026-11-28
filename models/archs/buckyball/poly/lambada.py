"""Prepare LAMBADA last-word targets, FP32 predictions, and the Poly decoder's empty KV state."""

from importlib import import_module
import multiprocessing
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer


def write_layout(prompt, destination, packs):
    destination.mkdir(parents=True, exist_ok=True)
    order = list(map(int, (prompt / "input-order.txt").read_text().split()))
    assert sorted(order) == list(range(len(order)))
    inputs = (prompt / "inputs.data").read_bytes()
    strides = [0] * (len(order) - 2)
    seeds = [b""] * len(strides)
    offset = 0
    input_index = 0
    outputs = []
    for line in (prompt / "sizes.txt").read_text().splitlines():
        kind, *dimensions = line.split()
        size, *axes = map(int, dimensions)
        if kind == "output":
            outputs.append((size, axes))
            continue
        index = order[input_index]
        if index >= 2:
            (capacity,) = axes
            data = inputs[offset : offset + size]
            stride = size // capacity
            pattern = data[:stride]
            if data != pattern * capacity:
                raise ValueError(
                    f"KV state {index - 2} has a position-dependent initial value"
                )
            strides[index - 2] = stride
            seeds[index - 2] = pattern
        input_index += 1
        offset += (size + 63) // 64 * 64
    assert input_index == len(order) and offset == len(inputs)
    values, indices, *updated = outputs
    candidates = values[0] // 4
    assert values == (4 * candidates, []) and indices == (8 * candidates, [])
    for stride, (size, axes) in zip(strides, updated, strict=True):
        (capacity,) = axes
        assert stride > 0 and size == stride * capacity
    np.asarray(
        [len(packs), len(strides), candidates, *strides, *order], dtype="<u8"
    ).tofile(destination / "cache-layout.data")
    (destination / "cache-seed.data").write_bytes(b"".join(seeds))
    for name in packs:
        source = prompt / name
        if not source.is_file():
            raise FileNotFoundError(source)
        link = destination / name
        link.unlink(missing_ok=True)
        link.symlink_to(source.relative_to(destination, walk_up=True))


def predict(text):
    context, _ = text.rsplit(" ", 1)
    ids = tokenizer(text).input_ids
    prefix = tokenizer(context).input_ids
    assert ids[: len(prefix)] == prefix
    target = ids[len(prefix) :]
    assert prefix and target and len(prefix) + len(target) <= 256
    with torch.inference_mode():
        logits = checkpoint(torch.tensor([ids[:-1]]), use_cache=False).logits[0]
    return prefix, target, logits[len(prefix) - 1 :].argmax(-1).tolist()


def set_worker_threads(threads):
    torch.set_num_threads(threads)
    threadpool_limits(limits=threads, user_api="blas")


if __name__ == "__main__":
    output = Path(sys.argv[1])
    dataset = output / "dataset"
    labels = dataset / "labels.data"
    labels.unlink(missing_ok=True)
    write_layout(output / "prompt", dataset / "model", sys.argv[2:])
    model = import_module(f"models.models.{output.name}.model")
    corpus = (
        Path(model.__file__).resolve().parents[2]
        / "datasets/lambada/en/test/en.parquet"
    )
    # Match the existing Poly Qwen3-0.6B evaluation cohort.
    texts = pq.read_table(corpus)["text"].to_pylist()[::8]
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(model.checkpoint)
    checkpoint = model.load().model.eval()
    cores = len(os.sched_getaffinity(0))
    workers = min(8, max(1, cores // 4))
    threads = min(4, cores // workers)
    rows = []
    with multiprocessing.get_context("fork").Pool(
        workers, initializer=set_worker_threads, initargs=(threads,)
    ) as pool:
        for row in pool.imap(predict, texts, chunksize=2):
            rows.append(row)
            if len(rows) % 64 == 0:
                print(
                    f"{output.name}: {len(rows)}/{len(texts)} reference passages",
                    flush=True,
                )
    lengths = np.asarray([(len(ctx), len(tgt)) for ctx, tgt, _ in rows], dtype="<i8")
    contexts = np.zeros((len(rows), lengths[:, 0].max()), dtype="<i8")
    targets = np.zeros((len(rows), lengths[:, 1].max()), dtype="<i8")
    reference = np.zeros_like(targets)
    for i, (ctx, tgt, predicted) in enumerate(rows):
        assert len(predicted) == len(tgt)
        contexts[i, : len(ctx)] = ctx
        targets[i, : len(tgt)] = tgt
        reference[i, : len(tgt)] = predicted
    contexts.tofile(dataset / "contexts.data")
    targets.tofile(dataset / "targets.data")
    reference.tofile(dataset / "reference.data")
    lengths.tofile(dataset / "lengths.data")
    completed = dataset / "labels.tmp"
    lengths[:, 1].tofile(completed)
    completed.replace(labels)
    print(f"Prepared {len(rows)} LAMBADA passages for {output.name}")

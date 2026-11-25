# ===- buddy-pytorch-dataset.py --------------------------------------------
#
# Qwen3-0.6B on poly: the dataset of the accuracy runs (dataset.cpp), every
# eighth of the 5153 passages of LAMBADA (OpenAI's English test set, 645;
# the accuracy run of the design stays within two hours), zero-shot: the
# model reads a passage without its last word and must give that word, each
# of its tokens the first maximum of the logits (greedy, the earlier tokens
# of the word given). In <output>/dataset/:
#   contexts.data   the tokens of each passage without its last word, padded
#                   to 248 int64s
#   targets.data    the tokens of the last word (with its space), padded to
#                   8 int64s
#   lengths.data    the counts of both, 2 int64s
#   reference.data  the tokens the FP32 checkpoint in PyTorch predicts at
#                   each target position, 8 int64s
#   labels.data     the count of target tokens, int64
#   model/          the packs and the program's input order and candidate
#                   counts, which every run reads from the model region
# The checkpoint's predictions are computed by 16 forked processes of 4
# threads.
#
#   buddy-pytorch-dataset.py <output>
#
# ===---------------------------------------------------------------------===

from importlib import import_module
from pathlib import Path
import multiprocessing
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

output = Path(sys.argv[1])
dataset = output / "dataset"
(dataset / "model").mkdir(parents=True, exist_ok=True)
model = import_module("models.models.qwen3-0-6b.model")
lambada = (
    Path(model.__file__).resolve().parents[2] / "datasets/lambada/en/test/en.parquet"
)
# One thread here: a forked worker would wait forever at the barrier of an
# OpenMP team this process had started.
torch.set_num_threads(1)
tokenizer = AutoTokenizer.from_pretrained(model.checkpoint)
checkpoint = AutoModelForCausalLM.from_pretrained(
    model.checkpoint, torch_dtype=torch.float32
).eval()

texts = pq.read_table(lambada)["text"].to_pylist()[::8]
samples = len(texts)


def predict(text):
    """The passage's tokens without its last word, the word's tokens and the
    checkpoint's prediction at each of their positions (the word's tokens but
    its last given)."""
    torch.set_num_threads(4)
    context, word = text.rsplit(" ", 1)
    context_ids = tokenizer(context).input_ids
    target_ids = tokenizer(" " + word, add_special_tokens=False).input_ids
    assert len(context_ids) <= 248 and len(target_ids) <= 8, text
    with torch.no_grad():
        logits = checkpoint(torch.tensor([context_ids + target_ids[:-1]])).logits[0]
    first = len(context_ids) - 1
    return (
        context_ids,
        target_ids,
        logits[first : first + len(target_ids)].argmax(-1).tolist(),
    )


contexts = np.zeros((samples, 248), np.int64)
targets = np.zeros((samples, 8), np.int64)
lengths = np.zeros((samples, 2), np.int64)
reference = np.zeros((samples, 8), np.int64)
# Forked: the workers share the checkpoint loaded above.
with multiprocessing.get_context("fork").Pool(16) as pool:
    for index, (context_ids, target_ids, predicted) in enumerate(
        pool.map(predict, texts)
    ):
        contexts[index, : len(context_ids)] = context_ids
        targets[index, : len(target_ids)] = target_ids
        lengths[index] = (len(context_ids), len(target_ids))
        reference[index, : len(target_ids)] = predicted

contexts.tofile(dataset / "contexts.data")
targets.tofile(dataset / "targets.data")
lengths.tofile(dataset / "lengths.data")
reference.tofile(dataset / "reference.data")
lengths[:, 1].copy().tofile(dataset / "labels.data")
for name, source in (
    ("pack-f32.data", "../../prompt/pack-f32.data"),
    ("pack-i8.data", "../../prompt/pack-i8.data"),
    ("pack-i32.data", "../../prompt/pack-i32.data"),
    ("input-order.txt", "../../input-order.txt"),
    ("candidates.txt", "../../candidates.txt"),
):
    link = dataset / "model" / name
    link.unlink(missing_ok=True)
    link.symlink_to(source)
correct = sum(
    bool((reference[i, : lengths[i, 1]] == targets[i, : lengths[i, 1]]).all())
    for i in range(samples)
)
print(f"{samples} LAMBADA passages; FP32 checkpoint {correct / samples:.4f} correct")

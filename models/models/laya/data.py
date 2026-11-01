"""Typed-decision records as the fine-tuned checkpoint reads them: every
question tokenized with its vocabulary to the 128 tokens of the graph, with
its option markers, question type and gold answer."""

from importlib import import_module
from pathlib import Path
import json

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

decisions = Path(__file__).resolve().parents[2] / "datasets/typed-decisions/all"
laya = import_module("models.models.laya.model")
tokenizer = AutoTokenizer.from_pretrained(laya.checkpoint / "typed-decisions/tokenizer")
qtypes = {"choice": 0, "score": 1, "noul": 2}


def sequence(state, question):
    """The tokens of one typed question as the checkpoint reads them, cut to
    the 128 tokens of the graph: [CLS] <type> question [SEP], a [MASK] marker
    before every option, [SEP] state [SEP]. The question and its options take
    at most 96 tokens, the state the rest. Returns the tokens and the marker
    positions."""
    kind, criteria = question["type"], question.get("criteria")
    if kind == "choice":
        if isinstance(criteria, list):
            criteria = {key: None for key in criteria}
        options = [
            key if not text else f"{key}: {text}" for key, text in criteria.items()
        ]
    elif kind == "score":
        options = [f"level {i}: {text}" for i, text in enumerate(criteria)]
    else:
        criteria = criteria or {}
        options = [
            "false: " + (criteria.get("false") or "no, the statement does not hold"),
            "true: " + (criteria.get("true") or "yes, the statement holds"),
        ]
    instructions = question["instructions"]
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions)
    head = tokenizer(f"{kind} question: {instructions}", add_special_tokens=False)
    pieces = [
        [tokenizer.mask_token_id]
        + tokenizer(" " + option, add_special_tokens=False)["input_ids"][:48]
        for option in options
    ]
    budget = 96 - sum(len(piece) for piece in pieces)
    if budget < 16:
        width = max(4, 80 // len(pieces))
        pieces = [piece[:width] for piece in pieces]
        budget = 96 - sum(len(piece) for piece in pieces)
    ids = [
        tokenizer.cls_token_id,
        *head["input_ids"][: max(8, budget)],
        tokenizer.sep_token_id,
    ]
    markers = []
    for piece in pieces:
        markers.append(len(ids))
        ids += piece
    ids.append(tokenizer.sep_token_id)
    ids += tokenizer(state, add_special_tokens=False)["input_ids"][: 127 - len(ids)]
    ids.append(tokenizer.sep_token_id)
    return ids[:128], [marker for marker in markers if marker < 128]


def load(first, count):
    """Every question of `count` typed-decision records from record `first`
    whose options fit the graph: tokens, mask, marker positions, valid
    markers, question types, and the index of the gold answer."""
    rows = pq.read_table(decisions / "train-00000-of-00001.parquet").slice(first, count)
    tokens, masks, positions, valids, types, labels = [], [], [], [], [], []
    for row in rows.to_pylist():
        gold = json.loads(row["gold"])
        for name, question in json.loads(row["questions"]).items():
            ids, markers = sequence(row["state"], question)
            options = 2 if question["type"] == "noul" else len(question["criteria"])
            if len(markers) != options or options > 16:
                continue
            label = gold[name]["label"]
            if question["type"] == "choice":
                label = list(question["criteria"]).index(label)
            elif question["type"] == "noul":
                label = int(label == "true")
            else:
                label = int(label)
            tokens.append(ids + [tokenizer.pad_token_id] * (128 - len(ids)))
            masks.append([1] * len(ids) + [0] * (128 - len(ids)))
            positions.append(markers + [0] * (16 - len(markers)))
            valids.append([True] * len(markers) + [False] * (16 - len(markers)))
            types.append(qtypes[question["type"]])
            labels.append(label)
    inputs = (
        torch.tensor(tokens),
        torch.tensor(masks),
        torch.tensor(positions),
        torch.tensor(valids),
        torch.tensor(types),
    )
    return inputs, torch.tensor(labels)

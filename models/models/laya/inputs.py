import hashlib
import json
import math

import torch

TYPES = {"choice": 0, "score": 1, "noul": 2}


def encode(tokenizer, request, length, options):
    kind = request["type"]
    qtype = TYPES[kind]

    def render(value):
        return (
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        )

    criteria = request.get("criteria") if kind == "noul" else request["criteria"]
    if kind == "choice":
        labels = list(criteria)
        texts = [
            str(key) if value is None or value == "" else f"{key}: {render(value)}"
            for key, value in criteria.items()
        ]
    elif kind == "score":
        labels = list(range(len(criteria)))
        texts = [f"level {i}: {render(text)}" for i, text in enumerate(criteria)]
    else:
        labels = [False, True]
        if criteria is None:
            criteria = {}
        texts = []
        for key, description in (
            ("false", "no, the statement does not hold"),
            ("true", "yes, the statement holds"),
        ):
            value = criteria.get(key)
            texts.append(
                key
                + ": "
                + (description if value is None or value == "" else render(value))
            )
    if not 1 <= len(texts) <= options:
        raise ValueError(f"expected 1..{options} options")

    def tokens(text):
        return tokenizer.encode(
            text.replace(tokenizer.mask_token, " "), add_special_tokens=False
        )

    ids = [
        tokenizer.cls_token_id,
        *tokens(f"{kind} question: {request['instructions']}"),
        tokenizer.sep_token_id,
    ]
    positions = []
    for text in texts:
        positions.append(len(ids))
        ids.extend([tokenizer.mask_token_id, *tokens(" " + text)])
    state = request["state"]
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    ids.extend([tokenizer.sep_token_id, *tokens(state), tokenizer.sep_token_id])
    if len(ids) > length:
        raise ValueError(
            f"request has {len(ids)} tokens; compiled capacity is {length}"
        )
    count = len(ids)
    return {
        "tokens": torch.tensor(
            [ids + [tokenizer.pad_token_id] * (length - count)], dtype=torch.int64
        ),
        "mask": torch.tensor([[1] * count + [0] * (length - count)], dtype=torch.int64),
        "positions": torch.tensor(
            [positions + [0] * (options - len(positions))], dtype=torch.int64
        ),
        "valid": torch.tensor(
            [[1] * len(positions) + [0] * (options - len(positions))], dtype=torch.int64
        ),
        "qtype": torch.tensor([qtype], dtype=torch.int64),
    }, labels


def prepare_inputs(directory, config_path, metadata):
    import struct
    import tomllib
    from transformers import AutoTokenizer

    settings = tomllib.loads(config_path.read_text())
    tiles = settings["tile_indices"]
    if tiles != [1]:
        raise ValueError("Laya execution selects its compute tile controller")
    tokenizer = AutoTokenizer.from_pretrained(
        directory / "tokenizer", local_files_only=True
    )
    (directory / "inputs").mkdir(exist_ok=True)
    resources = []
    labels_metadata = []
    reference_files = {}
    for name in settings["inputs"]:
        contents = (config_path.parent / name).read_bytes()
        reference_files[name] = hashlib.sha256(contents).hexdigest()
        requests = json.loads(contents)
        requests = requests if isinstance(requests, list) else [requests]
        for request in requests:
            encoded, labels = encode(
                tokenizer, request, metadata["sequence_length"], metadata["options"]
            )
            count = len(labels)
            bucket = (
                "2"
                if count <= 2
                else "3-5" if count <= 5 else "6-10" if count <= 10 else "11+"
            )
            key = f"{request['type']}:{bucket}"
            calibration = metadata["settings"]
            temperature = (
                calibration["temperature_by_options"][key]
                if key in calibration["temperature_by_options"]
                else calibration["temperature"][TYPES[request["type"]]]
            )
            if not math.isfinite(temperature):
                raise ValueError("non-finite calibration temperature")
            temperature = min(5.0, max(0.5, float(temperature)))
            packet = struct.pack(
                "<QQQQd",
                metadata["sequence_length"],
                metadata["options"],
                metadata["actions"],
                tiles[0],
                temperature,
            )
            for field in ("tokens", "mask", "positions", "valid", "qtype"):
                value = encoded[field]
                packet += struct.pack(f"<{value.numel()}q", *value.flatten().tolist())
            relative = f"inputs/request-{len(resources)}.bin"
            (directory / relative).write_bytes(packet)
            resources.append(relative)
            labels_metadata.append({"request": request, "labels": labels})
    (directory / "inputs/labels.json").write_text(
        json.dumps(labels_metadata, ensure_ascii=False)
    )
    return (
        resources,
        {key: settings[key] for key in ("inputs", "tile_indices")},
        reference_files,
    )

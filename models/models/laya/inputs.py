import json

import numpy as np

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
        "tokens": np.array(
            [ids + [tokenizer.pad_token_id] * (length - count)], dtype=np.int64
        ),
        "mask": np.array([[1] * count + [0] * (length - count)], dtype=np.int64),
        "positions": np.array(
            [positions + [0] * (options - len(positions))], dtype=np.int64
        ),
        "valid": np.array(
            [[1] * len(positions) + [0] * (options - len(positions))], dtype=np.int64
        ),
        "qtype": np.array([qtype], dtype=np.int64),
    }, labels


def decode(logits, action, request, labels, config):
    kind = request["type"]
    count = len(labels)
    bucket = (
        "2" if count <= 2 else "3-5" if count <= 5 else "6-10" if count <= 10 else "11+"
    )
    temperatures = config["temperature_by_options"]
    key = f"{kind}:{bucket}"
    temperature = (
        temperatures[key] if key in temperatures else config["temperature"][TYPES[kind]]
    )
    if not np.isfinite(temperature):
        raise ValueError("non-finite calibration temperature")
    temperature = min(5.0, max(0.5, float(temperature)))
    values = logits[:count].astype(np.float64) / temperature
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    values = action.astype(np.float64)
    actions = np.exp(values - values.max())
    actions /= actions.sum()
    answer = labels[int(probabilities.argmax())]
    if kind == "score":
        answer = float(probabilities @ np.arange(count))
    elif kind == "noul":
        answer = float(probabilities[1])
    return {
        "type": kind,
        "answer": answer,
        "probabilities": probabilities.tolist(),
        "action_probabilities": actions.tolist(),
    }

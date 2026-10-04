import argparse
import json
import math
import re
import struct
from pathlib import Path


def read_logits(path, labels):
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text())
    text = "\n".join(re.sub(r"^\[[^\]\n]+\]\s*", "", line) for line in text.splitlines())
    blocks = re.findall(r"LOGITS_F32_BEGIN(.*?)LOGITS_F32_END", text, re.S)
    if len(blocks) != 1:
        raise ValueError("expected exactly one complete logits block")
    words = []
    for line in blocks[0].splitlines():
        if not line.strip():
            continue
        word = line.split()[-1]
        if not re.fullmatch(r"[0-9a-fA-F]{1,8}", word):
            raise ValueError(f"invalid float bits: {word}")
        words.append(int(word, 16))
    if len(words) != labels:
        raise ValueError("logits count differs from model metadata")
    values = [struct.unpack("<f", struct.pack("<I", word))[0] for word in words]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite model output")
    return words, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uart", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    words, values = read_logits(args.uart, metadata["num_labels"])
    golden, _ = read_logits(args.golden, metadata["num_labels"])
    if words != golden:
        raise ValueError(f"hardware output differs from BEMU: {words} != {golden}")
    reference = metadata["validation_inputs"][0]["reference_logits"]
    cosine = sum(a*b for a,b in zip(values, reference)) / math.sqrt(
        sum(a*a for a in values) * sum(b*b for b in reference))
    print(json.dumps({"backend_match": "all float bits equal", "logits": values,
        "reference_logits": reference, "cosine": cosine,
        "max_abs_error": max(abs(a-b) for a,b in zip(values, reference)),
        "label": metadata["labels"][max(range(len(values)), key=values.__getitem__)]}))


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description="Compare compiled MLP-Mixer logits with PyTorch")
parser.add_argument("reference", type=Path)
parser.add_argument("actual", type=Path)
parser.add_argument("--uart", action="store_true")
args = parser.parse_args()
reference = np.fromfile(args.reference, dtype="<f4")
if args.uart:
    text = args.actual.read_text()
    words = text.split("LOGITS_F32_BEGIN\n")[1].split("LOGITS_F32_END")[0].split()
    actual = np.asarray([int(word, 16) for word in words], dtype="<u4").view("<f4")
else:
    actual = np.fromfile(args.actual, dtype="<f4")
if reference.shape != (1000,) or actual.shape != reference.shape:
    raise ValueError("expected 1000 ImageNet logits in each output")
if not np.isfinite(reference).all() or not np.isfinite(actual).all():
    raise ValueError("non-finite inference output")
error = actual - reference
expected = np.argsort(reference)[-5:][::-1].tolist()
observed = np.argsort(actual)[-5:][::-1].tolist()
cosine = float(np.dot(actual, reference) / (np.linalg.norm(actual) * np.linalg.norm(reference)))
print(json.dumps({"reference_top5": expected, "actual_top5": observed,
                  "top1_match": expected[0] == observed[0],
                  "top5_overlap": len(set(expected) & set(observed)),
                  "cosine_similarity": cosine,
                  "max_abs_error": float(np.abs(error).max()),
                  "rmse": float(np.sqrt(np.mean(error ** 2)))}, indent=2))
if observed[0] != expected[0] or cosine < 0.99:
    raise SystemExit("MLP-Mixer numerical gate failed: require matching Top-1 and cosine >= 0.99")

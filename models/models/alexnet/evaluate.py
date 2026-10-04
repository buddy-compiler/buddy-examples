import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description="Compare compiled AlexNet logits with PyTorch")
parser.add_argument("reference", type=Path)
parser.add_argument("actual", type=Path)
args = parser.parse_args()
reference = np.fromfile(args.reference, dtype="<f4")
actual = np.fromfile(args.actual, dtype="<f4")
if reference.shape != (1000,) or actual.shape != reference.shape:
    raise ValueError("expected 1000 ImageNet logits in each output")
if not np.isfinite(reference).all() or not np.isfinite(actual).all():
    raise ValueError("non-finite inference output")
error = actual - reference
expected = np.argsort(reference)[-5:][::-1].tolist()
observed = np.argsort(actual)[-5:][::-1].tolist()
print(json.dumps({"reference_top5": expected, "actual_top5": observed,
                  "top1_match": expected[0] == observed[0],
                  "top5_overlap": len(set(expected) & set(observed)),
                  "cosine_similarity": float(np.dot(actual, reference) / (np.linalg.norm(actual) * np.linalg.norm(reference))),
                  "max_abs_error": float(np.abs(error).max()),
                  "rmse": float(np.sqrt(np.mean(error ** 2)))}, indent=2))

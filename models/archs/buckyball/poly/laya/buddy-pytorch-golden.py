# ===- buddy-pytorch-golden.py --------------------------------------------
#
# Laya on poly: the golden data of baremetal.cpp. One held-out typed
# decision (input.data, shared by baremetal and Linux), the choice and action
# scores of the quantized model in PyTorch on it (expected.data,
# frontend/reference.py), and its choice, action and label with the choice of
# the FP32 model (reference.json).
#
# ===---------------------------------------------------------------------===

from importlib import import_module
from pathlib import Path
import sys
import json

import numpy as np
import torch

torch.set_num_threads(16)
from torch.nn.attention import SDPBackend, sdpa_kernel

from frontend.reference import evaluate, quantized_forward

model_dir = Path(__file__).resolve().parent
output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)

laya = import_module("models.models.laya.model")
decisions = import_module("models.models.laya.data")
# The recipe runs the checkpoint fine-tuned on typed decisions.
laya.checkpoint = laya.checkpoint / "typed-decisions"
model = laya.load()

sample, labels = decisions.load(64, 1)
sample = tuple(value[:1] for value in sample)
(output / "input.data").write_bytes(
    b"".join(value.numpy().tobytes() for value in sample)
)
# The attention of the head runs as its math, as the importer traces it.
with sdpa_kernel(SDPBackend.MATH):
    baseline, expected = evaluate(model, sample)
np.concatenate([value.numpy().reshape(-1) for value in expected]).tofile(
    output / "expected.data"
)
(output / "reference.json").write_text(
    json.dumps(
        {
            "fp32_choice": baseline[0].argmax(-1).item(),
            "choice": expected[0].argmax(-1).item(),
            "action": expected[1].argmax(-1).item(),
            "label": labels[0].item(),
        }
    )
    + "\n"
)

# ===----------------------------------------------------------------------===
# The dataset of the accuracy runs (dataset.cpp, from the model region): the
# 200 held-out decisions of buddy-pytorch-calibrate.py, each an Input, the
# choice and action scores of the same quantized model on each and their
# labels; under model/, the packs the program reads from the region.
# ===----------------------------------------------------------------------===
# The single-sample evaluation prepared this model's weights for MXINT8.
tests, test_labels = decisions.load(200, 200)
# Run one question at a time, as the program does.
with sdpa_kernel(SDPBackend.MATH):
    results = [
        quantized_forward(model, tuple(value[index : index + 1] for value in tests))
        for index in range(len(test_labels))
    ]
choices = torch.cat([choice for choice, _ in results])
actions = torch.cat([action for _, action in results])
dataset = output / "dataset"
(dataset / "model/ddr").mkdir(parents=True, exist_ok=True)
(dataset / "inputs.data").write_bytes(
    b"".join(
        value[index].numpy().tobytes()
        for index in range(len(test_labels))
        for value in tests
    )
)
np.concatenate([choices.numpy(), actions.numpy()], axis=1).astype(np.float32).tofile(
    dataset / "reference.data"
)
test_labels.numpy().astype(np.int64).tofile(dataset / "labels.data")
for name in ("pack-f32.data", "pack-i8.data", "pack-i32.data"):
    link = dataset / "model/ddr" / name
    link.unlink(missing_ok=True)
    link.symlink_to(f"../../../sample/{name}")

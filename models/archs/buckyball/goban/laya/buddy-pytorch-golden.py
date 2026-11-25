# ===- buddy-pytorch-golden.py --------------------------------------------
#
# Laya on goban: the golden data of baremetal.cpp. One held-out typed
# decision (input.data, shared by baremetal and Linux), the choice and action
# scores in PyTorch on it of the smoothed model with the calibrated INT8
# operands of calibration.json (expected.data, frontend/reference.py), and
# its choice, action and label (reference.json).
#
#   buddy-pytorch-golden.py <output> <libnewlib-reference.so>   the newlib
#                                       library of the harts' sinf and cosf
#
# ===---------------------------------------------------------------------===

from importlib import import_module
from pathlib import Path
import sys
import json

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from frontend.quant.graph import smooth_inputs
from frontend.reference import evaluate

# Eight threads: many threads on one process are slower on a loaded host.
torch.set_num_threads(8)


model_dir = Path(__file__).resolve().parent
output = Path(sys.argv[1])
newlib = sys.argv[2]
output.mkdir(parents=True, exist_ok=True)

laya = import_module("models.models.laya.model")
decisions = import_module("models.models.laya.data")
# The recipe runs the checkpoint fine-tuned on typed decisions.
laya.checkpoint = laya.checkpoint / "typed-decisions"
model = laya.load()
# The smoothing of buddy-pytorch-calibrate.py, as the importer applies it.
calibration = json.loads((model_dir / "output/calibration.json").read_text())
smooth_inputs(model, calibration["smooth"])

sample, labels = decisions.load(64, 1)
sample = tuple(value[:1] for value in sample)
(output / "input.data").write_bytes(
    b"".join(value.numpy().tobytes() for value in sample)
)
# The attention of the head runs as its math, as the importer traces it.
with sdpa_kernel(SDPBackend.MATH):
    expected = evaluate(model, calibration, sample, newlib)
np.concatenate([value.numpy().reshape(-1) for value in expected]).tofile(
    output / "expected.data"
)
(output / "reference.json").write_text(
    json.dumps(
        {
            "choice": expected[0].argmax(-1).item(),
            "action": expected[1].argmax(-1).item(),
            "label": labels[0].item(),
        }
    )
    + "\n"
)

# ===----------------------------------------------------------------------===
# The dataset of the accuracy runs (dataset.cpp, from the model region): the
# 200 held-out decisions of buddy-pytorch-calibrate.py, each a laya::Input,
# the choice and action scores of the same reference on each and their
# labels; under model/, the weights the program reads from the region.
# ===----------------------------------------------------------------------===
tests, labels = decisions.load(200, 200)
# One question at a time, as the program runs them: evaluate() of a batch
# differs from it (by a tenth on some questions).
with sdpa_kernel(SDPBackend.MATH):
    results = [
        evaluate(
            model,
            calibration,
            tuple(value[index : index + 1] for value in tests),
            newlib,
        )
        for index in range(len(labels))
    ]
choices = torch.cat([choice for choice, _ in results])
actions = torch.cat([action for _, action in results])
dataset = output / "dataset"
(dataset / "model/ddr").mkdir(parents=True, exist_ok=True)
(dataset / "inputs.data").write_bytes(
    b"".join(
        value[index].numpy().tobytes()
        for index in range(len(labels))
        for value in tests
    )
)
np.concatenate([choices.numpy(), actions.numpy()], axis=1).astype(np.float32).tofile(
    dataset / "reference.data"
)
labels.numpy().astype(np.int64).tofile(dataset / "labels.data")
for index in range(3):
    link = dataset / f"model/ddr/arg{index}.data"
    link.unlink(missing_ok=True)
    link.symlink_to(f"../../../arg{index}.data")

"""Check the integer MXINT8 GELU candidate against Laya's existing accuracy gate."""

from pathlib import Path
import runpy
import torch

from frontend.integer_activations import IntegerActivations

torch.set_num_threads(16)
reference = runpy.run_path(str(Path(__file__).with_name("buddy-pytorch-calibrate.py")))
fp32, baseline = reference["evaluate"](False)
print(
    f"FP32 baseline accuracy {fp32:.2f}% on {len(reference['labels'])} decisions",
    flush=True,
)
mode = IntegerActivations()
logits = []
with torch.no_grad(), reference["Products"](), reference["Norms"](), mode:
    for index, batch in enumerate(
        zip(*(value.split(16) for value in reference["tests"]))
    ):
        logits.append(reference["model"](*batch)[0])
        if index % 8 == 0:
            print(f"Integer activation calibration batch {index + 1}", flush=True)
assert mode.calls > 0, "Integer GELU candidate did not execute"
assert mode.gated_calls > 0, "Integer gated GELU candidate did not execute"
logits = torch.cat(logits)
accuracy = (logits.argmax(1) == reference["labels"]).float().mean().item() * 100
valid = reference["tests"][3]
error = ((logits - baseline)[valid].norm() / baseline[valid].norm()).item()
print(
    f"Integer MXINT8 GELU accuracy {accuracy:.2f}% on {len(reference['labels'])} decisions, "
    f"marker logits relative L2 error {error:.4f}, GELU calls {mode.calls}, gated calls {mode.gated_calls}"
)
if fp32 - accuracy > 1:
    raise RuntimeError("Integer GELU accuracy loss exceeds one percentage point")

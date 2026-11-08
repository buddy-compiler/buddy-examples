# ===- buddy-pytorch-calibrate.py -----------------------------------------
#
# SmolVLA quantization check on piplus: the program of the vision, prefix
# and step graphs as the quantized graphs compute it — every matrix product
# with MXINT8 operands, zero-padded along K to a multiple of 32, Linear
# weights as two MXINT8 terms — rejected
# when its actions on SO100 frames drift from the FP32 policy. Every
# activation scale is computed online, so nothing is calibrated:
# calibration.json records the accepted error.
#
# ===---------------------------------------------------------------------------

from importlib import import_module
from pathlib import Path
import json

import torch

from frontend.graphs import Prefix, Step, Vision
from frontend.program import run
from frontend.quant.graph import doubled
from frontend.quant.reference import Products, bf16, mxint8

torch.set_num_threads(4)

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
smolvla = import_module("models.models.smolvla-base.model")
policy = smolvla.load()
# The vision encoder attends through explicit products, as the importer.
policy.flow.vlm_with_expert.get_vlm_model().vision_model.config._attn_implementation = (
    "eager"
)
flow = policy.flow
vision, prefix, step = Vision(flow).eval(), Prefix(flow).eval(), Step(flow).eval()
steps = flow.config.num_steps

# The evaluation inputs: the 100th frame of four SO100 pick-and-place
# episodes.
observation = import_module("models.models.smolvla-base.observation")
evaluation = [observation.frame(episode) for episode in (40, 43, 46, 49)]


with torch.no_grad():
    fp32 = torch.stack([policy(*inputs) for inputs in evaluation])
    with Products(bf16, set()):
        reference = torch.stack(
            [run(vision, prefix, step, steps, *inputs) for inputs in evaluation]
        )
    with Products(mxint8, doubled(flow)):
        quantized = torch.stack(
            [run(vision, prefix, step, steps, *inputs) for inputs in evaluation]
        )
reference = ((reference - fp32).norm() / fp32.norm()).item()
print(f"BF16 products: relative L2 error {reference:.4f} against FP32")

# ===----------------------------------------------------------------------===
# The quantized program: every product with MXINT8 operands, weights packed
# along K and activations quantized online, K zero-padded to a multiple of
# 32; Linear weights use two MXINT8
# terms; biases remain FP32. Rejected when the relative L2 error of the actions
# exceeds 0.05.
# ===----------------------------------------------------------------------===
error = ((quantized - fp32).norm() / fp32.norm()).item()
print(f"Quantized actions: relative L2 error {error:.4f} against FP32")
if error > 0.05:
    raise RuntimeError(
        f"Quantized actions drift from FP32: relative L2 error {error:.4f}"
    )
(output / "calibration.json").write_text(json.dumps({"error": error}, indent=2) + "\n")

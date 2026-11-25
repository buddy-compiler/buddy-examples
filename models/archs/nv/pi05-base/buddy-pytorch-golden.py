# ===- buddy-pytorch-golden.py --------------------------------------------
#
# pi0.5: the test case of the runner, fixed random inputs and the actions
# PyTorch samples from them (output/testcase/).
#
# ===---------------------------------------------------------------------------

from importlib import import_module
from pathlib import Path

import torch

model_dir = Path(__file__).resolve().parent
testcase = model_dir / "output" / "testcase"
testcase.mkdir(parents=True, exist_ok=True)

pi05 = import_module("models.models.pi05-base.model")
model = pi05.load()
# The vision encoder attends through explicit products, as the importer
# imports it.
model.flow.paligemma_with_expert.paligemma.model.vision_tower.config._attn_implementation = (
    "eager"
)

example = pi05.inputs()
torch.manual_seed(0)
images = torch.rand(example[0].shape) * 2 - 1
tokens = torch.randint(0, 1000, example[1].shape)
masks = example[2]
noise = torch.randn(example[3].shape)
with torch.no_grad():
    actions = model(images, tokens, masks, noise)
images.numpy().tofile(testcase / "images.data")
tokens.numpy().tofile(testcase / "tokens.data")
masks.numpy().tofile(testcase / "masks.data")
noise.numpy().tofile(testcase / "noise.data")
actions.numpy().tofile(testcase / "actions.data")

# ===- buddy-pytorch-golden.py --------------------------------------------
#
# SmolVLA: the test case of the runner, fixed random inputs and the actions
# PyTorch samples from them (output/testcase/).
#
# ===---------------------------------------------------------------------------

from importlib import import_module
from pathlib import Path

import torch

model_dir = Path(__file__).resolve().parent
testcase = model_dir / "output" / "testcase"
testcase.mkdir(parents=True, exist_ok=True)

smolvla = import_module("models.models.smolvla-base.model")
model = smolvla.load()
# The vision encoder attends through explicit products, as the importer
# imports it.
model.flow.vlm_with_expert.get_vlm_model().vision_model.config._attn_implementation = (
    "eager"
)

example = smolvla.inputs()
torch.manual_seed(0)
images = torch.rand(example[0].shape) * 2 - 1
lang_tokens = torch.randint(0, 1000, example[1].shape)
lang_masks = example[2]
state = torch.randn(example[3].shape)
noise = torch.randn(example[4].shape)
with torch.no_grad():
    actions = model(images, lang_tokens, lang_masks, state, noise)
images.numpy().tofile(testcase / "images.data")
lang_tokens.numpy().tofile(testcase / "lang_tokens.data")
lang_masks.numpy().tofile(testcase / "lang_masks.data")
state.numpy().tofile(testcase / "state.data")
noise.numpy().tofile(testcase / "noise.data")
actions.numpy().tofile(testcase / "actions.data")

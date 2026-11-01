"""SO100 pick-and-place frames as smolvla-base takes them: the 100th frame of
an episode, prepared as the policy prepares it: the two cameras resized with
padding to 512 x 512 in [-1, 1] and an empty third camera; the state
normalized with the mean and deviation of the dataset and padded to 32; the
task ending in a newline, tokenized with the SmolVLM2 vocabulary to 48
tokens; the sampling noise drawn with the episode as its seed."""

from importlib import import_module
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
from transformers import AutoTokenizer

smolvla = import_module("models.models.smolvla-base.model")
root = Path(__file__).resolve().parents[2] / "datasets/svla-so100-pickplace"
dataset = LeRobotDataset("lerobot/svla_so100_pickplace", root=root)
statistics = json.loads((root / "meta/stats.json").read_text())["observation.state"]
tokenizer = AutoTokenizer.from_pretrained(smolvla.config.vlm_model_name)
starts = dataset.meta.episodes["dataset_from_index"]


def frame(episode):
    """The images, prompt tokens, their mask, state and noise of the 100th
    frame of `episode`."""
    sample = dataset[starts[episode] + 100]
    cameras = [
        resize_with_pad(sample[key][None], 512, 512, pad_value=0) * 2 - 1
        for key in ("observation.images.top", "observation.images.wrist")
    ]
    cameras.append(-torch.ones_like(cameras[0]))
    mean = torch.tensor(statistics["mean"])
    std = torch.tensor(statistics["std"])
    state = F.pad((sample["observation.state"] - mean) / std, (0, 26))[None]
    tokens = tokenizer(
        sample["task"].strip() + "\n",
        max_length=48,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    noise = torch.randn(
        smolvla.inputs()[4].shape, generator=torch.Generator().manual_seed(episode)
    )
    return torch.stack(cameras), tokens.input_ids, tokens.attention_mask, state, noise

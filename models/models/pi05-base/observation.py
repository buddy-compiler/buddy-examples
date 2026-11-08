"""SO100 pick-and-place frames as pi05-base takes them: the 100th frame of an
episode, prepared as the policy prepares it: the two cameras resized with
padding to 224 x 224 in [-1, 1] and an empty third camera; the state
normalized to [-1, 1] with the range of the dataset and written into the
prompt in 256 bins, tokenized with the Gemma vocabulary of PaliGemma to 200
tokens; the sampling noise drawn with the episode as its seed."""

from importlib import import_module
from pathlib import Path
import json

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
from transformers import AutoTokenizer

pi05 = import_module("models.models.pi05-base.model")
root = Path(__file__).resolve().parents[2] / "datasets/svla-so100-pickplace"
dataset = LeRobotDataset("lerobot/svla_so100_pickplace", root=root)
statistics = json.loads((root / "meta/stats.json").read_text())["observation.state"]
tokenizer = AutoTokenizer.from_pretrained(pi05.checkpoint.parent / "gemma-tokenizer")
starts = dataset.meta.episodes["dataset_from_index"]


def frame(episode):
    """The images, prompt tokens, their mask and noise of the 100th frame of
    `episode`."""
    sample = dataset[starts[episode] + 100]
    cameras = [
        resize_with_pad(sample[key][None], 224, 224, pad_value=0) * 2 - 1
        for key in ("observation.images.top", "observation.images.wrist")
    ]
    cameras.append(-torch.ones_like(cameras[0]))
    low = torch.tensor(statistics["min"])
    high = torch.tensor(statistics["max"])
    state = (2 * (sample["observation.state"] - low) / (high - low) - 1).clamp(-1, 1)
    bins = np.digitize(state.numpy(), bins=np.linspace(-1, 1, 257)[:-1]) - 1
    task = sample["task"].strip().replace("_", " ").replace("\n", " ")
    prompt = f"Task: {task}, State: {' '.join(map(str, bins))};\nAction: "
    tokens = tokenizer(
        prompt,
        max_length=200,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    noise = torch.randn(
        pi05.inputs()[3].shape, generator=torch.Generator().manual_seed(episode)
    )
    return torch.stack(cameras), tokens.input_ids, tokens.attention_mask, noise

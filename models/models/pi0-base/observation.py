"""SO100 pick-and-place frames as pi0-base takes them: the 100th frame of an
episode, prepared as the policy prepares it: the two cameras resized with
padding to 224 x 224 in [-1, 1] and an empty third camera; the task as the
prompt, tokenized with the Gemma vocabulary of PaliGemma to 48 tokens; the
state normalized to [-1, 1] with the range of the dataset and padded to the
32 values of the policy; the sampling noise drawn with the episode as its
seed."""

from importlib import import_module
from pathlib import Path
import json

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
from transformers import AutoTokenizer

pi0 = import_module("models.models.pi0-base.model")
root = Path(__file__).resolve().parents[2] / "datasets/svla-so100-pickplace"
dataset = LeRobotDataset("lerobot/svla_so100_pickplace", root=root)
statistics = json.loads((root / "meta/stats.json").read_text())["observation.state"]
tokenizer = AutoTokenizer.from_pretrained(pi0.checkpoint.parent / "gemma-tokenizer")
starts = dataset.meta.episodes["dataset_from_index"]


def frame(episode):
    """The images, prompt tokens, their mask, state and noise of the 100th
    frame of `episode`."""
    sample = dataset[starts[episode] + 100]
    cameras = [
        resize_with_pad(sample[key][None], 224, 224, pad_value=0) * 2 - 1
        for key in ("observation.images.top", "observation.images.wrist")
    ]
    cameras.append(-torch.ones_like(cameras[0]))
    low = torch.tensor(statistics["min"])
    high = torch.tensor(statistics["max"])
    state = (2 * (sample["observation.state"] - low) / (high - low) - 1).clamp(-1, 1)
    state = torch.nn.functional.pad(
        state, (0, pi0.config.max_state_dim - state.shape[0])
    )
    task = sample["task"].strip().replace("_", " ").replace("\n", " ")
    tokens = tokenizer(
        f"{task}\n",
        max_length=pi0.config.tokenizer_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    noise = torch.randn(
        pi0.inputs()[4].shape, generator=torch.Generator().manual_seed(episode)
    )
    return (
        torch.stack(cameras),
        tokens.input_ids,
        tokens.attention_mask,
        state[None],
        noise,
    )

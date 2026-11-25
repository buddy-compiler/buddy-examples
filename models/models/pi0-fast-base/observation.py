"""SO100 pick-and-place frames as pi0-fast-base takes them: the 100th frame of an
episode, prepared as the policy prepares it: the two cameras resized with
padding to 224 x 224 in [-1, 1] and an empty third camera; the state
normalized to [-1, 1] with the range of the dataset and written into the
prompt in 256 bins, tokenized with the Gemma vocabulary of PaliGemma and
followed by "Action:", 200 tokens."""

from importlib import import_module
from pathlib import Path
import json

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
from transformers import AutoTokenizer

pi0_fast = import_module("models.models.pi0-fast-base.model")
root = Path(__file__).resolve().parents[2] / "datasets/svla-so100-pickplace"
dataset = LeRobotDataset("lerobot/svla_so100_pickplace", root=root)
statistics = json.loads((root / "meta/stats.json").read_text())["observation.state"]
# As the policy's preprocessor loads it (TokenizerProcessorStep): the
# tokenizer's own BOS before the prompt and no EOS after it, padding after
# the prompt.
tokenizer = AutoTokenizer.from_pretrained(
    pi0_fast.checkpoint.parent / "gemma-tokenizer",
    padding_side="right",
)
starts = dataset.meta.episodes["dataset_from_index"]


def frame(episode):
    """The images, prompt tokens and their mask of the 100th frame of
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
    prompt = f"Task: {task}, State: {' '.join(map(str, bins))};\n"
    # The prompt, then "Action:" (openpi's sequence: the model continues it
    # with the space of "Action: " (model.Start), the FAST tokens and an
    # EOS), padded to the length of the policy.
    ids = (
        tokenizer(prompt).input_ids
        + tokenizer.encode("Action: ", add_special_tokens=False)[:-1]
    )
    length = pi0_fast.config.tokenizer_max_length
    tokens = torch.full((1, length), tokenizer.pad_token_id, dtype=torch.int64)
    tokens[0, : len(ids)] = torch.tensor(ids)
    masks = torch.zeros((1, length), dtype=torch.int64)
    masks[0, : len(ids)] = 1
    return torch.stack(cameras), tokens, masks

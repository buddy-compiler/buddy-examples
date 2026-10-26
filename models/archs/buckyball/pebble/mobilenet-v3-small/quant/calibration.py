from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class ImageNetCalibration(Dataset):
    def __init__(self):
        directory = Path(__file__).resolve().parents[5] / "datasets/imagenet-256/data"
        self.training = load_dataset(
            "parquet", data_files=str(directory / "train-*.parquet"), split="train"
        )

    def __len__(self):
        return len(self.training)

    def __getitem__(self, index):
        sample = self.training[index]
        pixels = np.asarray(
            sample["image"].convert("RGB"), dtype=np.float32
        ) / np.float32(255)
        height, width = pixels.shape[:2]
        y = np.arange(224, dtype=np.float32) * (np.float32(height) / 224)
        x = np.arange(224, dtype=np.float32) * (np.float32(width) / 224)
        yl, yh = np.floor(y).astype(int), np.minimum(np.ceil(y).astype(int), height - 1)
        xl, xh = np.floor(x).astype(int), np.minimum(np.ceil(x).astype(int), width - 1)
        wy, wx = (y - np.floor(y))[:, None, None], (x - np.floor(x))[None, :, None]
        a, b = pixels[yl[:, None], xl], pixels[yh[:, None], xl]
        c, d = pixels[yl[:, None], xh], pixels[yh[:, None], xh]
        # Match the current guest DIP resize, including its corner ordering.
        pixels = (a * ((1 - wx) * (1 - wy)) + b * (wx * (1 - wy))) + (
            c * (wy * (1 - wx)) + d * (wx * wy)
        )
        pixels = (
            pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return torch.from_numpy(pixels.transpose(2, 0, 1).copy()), sample["label"]

from pathlib import Path

import numpy as np
import torch
from PIL import Image
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
        image = sample["image"].convert("RGB")
        width, height = image.size
        target = (
            (256, int(256 * height / width))
            if width <= height
            else (int(256 * width / height), 256)
        )
        image = image.resize(target, Image.Resampling.BICUBIC)
        left, top = round((image.width - 224) / 2), round((image.height - 224) / 2)
        pixels = np.asarray(
            image.crop((left, top, left + 224, top + 224)), dtype=np.float32
        ) / np.float32(127.5) - np.float32(1)
        pixels = torch.from_numpy(pixels.transpose(2, 0, 1).copy())
        patches = (
            pixels.reshape(3, 14, 16, 14, 16)
            .permute(1, 3, 0, 2, 4)
            .reshape(196, 768)
            .contiguous()
        )
        return patches, sample["label"]

from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class EmotionCalibration(Dataset):
    def __init__(self, length):
        self.length = length
        root = Path(__file__).resolve().parents[5]
        self.data = load_dataset(
            "parquet",
            data_files=str(root / "datasets/emotion/split/train-*.parquet"),
            split="train",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            root / "models/bert-base-uncased-emotion/output/bert-base-uncased-emotion"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        encoded = self.tokenizer(
            sample["text"],
            max_length=self.length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {name: value[0] for name, value in encoded.items()}
        inputs["position_ids"] = torch.arange(self.length, dtype=torch.int64)
        return inputs, sample["label"]

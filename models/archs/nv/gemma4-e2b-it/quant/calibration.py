from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class WikiTextCalibration(Dataset):
    def __init__(self, tokenizer, length):
        directory = (
            Path(__file__).resolve().parents[4] / "datasets/wikitext/wikitext-2-raw-v1"
        )
        training = load_dataset(
            "parquet", data_files=str(directory / "train-*.parquet"), split="train"
        )
        self.tokens = torch.tensor(
            tokenizer(
                "\n\n".join(training["text"]),
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"],
            dtype=torch.int64,
        )
        self.length = length

    def __len__(self):
        return self.tokens.numel() // self.length

    def __getitem__(self, index):
        return self.tokens[index * self.length : (index + 1) * self.length]

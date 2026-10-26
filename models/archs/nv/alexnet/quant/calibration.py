from pathlib import Path
import cv2
import numpy as np

from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms


class ImageNetCalibration(Dataset):
    def __init__(self):
        directory = Path(__file__).resolve().parents[4] / "datasets/imagenet-256/data"
        self.training = load_dataset(
            "parquet", data_files=str(directory / "train-*.parquet"), split="train"
        )
        if len(self.training) != 1281167:
            raise ValueError("Expected the complete ImageNet-256 training split")
        self.transform = transforms.Compose(
            [
                lambda image: cv2.resize(
                    np.asarray(image), (224, 224), interpolation=cv2.INTER_LINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self.training)

    def __getitem__(self, index):
        sample = self.training[index]
        return self.transform(sample["image"].convert("RGB")), sample["label"]

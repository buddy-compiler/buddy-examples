from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from ultralytics.data.augment import LetterBox


class CocoCalibration(Dataset):
    def __init__(self):
        directory = (
            Path(__file__).resolve().parents[4] / "datasets/coco128/images/train2017"
        )
        self.images = sorted(directory.glob("*.jpg"))
        self.letterbox = LetterBox(new_shape=(640, 640), auto=False)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert("RGB"))
        image = self.letterbox(image=image)
        return torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0, 0

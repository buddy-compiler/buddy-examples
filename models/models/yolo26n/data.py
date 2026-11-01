"""COCO128 training images as the graph takes them: letterboxed to 640 x 640
in [0, 1], with their boxes in the letterboxed pixels."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics.data.augment import LetterBox

coco = Path(__file__).resolve().parents[2] / "datasets/coco128"
paths = sorted((coco / "images/train2017").glob("*.jpg"))


def load(paths):
    """The images letterboxed to 640x640 as the graph takes them, and their
    boxes as [class, x1, y1, x2, y2] in the letterboxed pixels."""
    images, labels = [], []
    for path in paths:
        image = np.asarray(Image.open(path).convert("RGB"))
        height, width = image.shape[:2]
        boxed = LetterBox((640, 640))(image=image)
        images.append(torch.from_numpy(boxed).permute(2, 0, 1).float() / 255)
        ratio = min(640 / height, 640 / width)
        pad_x, pad_y = (640 - width * ratio) / 2, (640 - height * ratio) / 2
        label = coco / "labels/train2017" / (path.stem + ".txt")
        rows = np.loadtxt(label, ndmin=2) if label.exists() else np.zeros((0, 5))
        boxes = torch.tensor(rows, dtype=torch.float32)
        x, y = boxes[:, 1] * width, boxes[:, 2] * height
        w, h = boxes[:, 3] * width, boxes[:, 4] * height
        labels.append(
            torch.stack(
                [
                    boxes[:, 0],
                    (x - w / 2) * ratio + pad_x,
                    (y - h / 2) * ratio + pad_y,
                    (x + w / 2) * ratio + pad_x,
                    (y + h / 2) * ratio + pad_y,
                ],
                1,
            )
        )
    return torch.stack(images), labels

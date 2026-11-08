"""ImageNet-256 images as the checkpoint takes them: bicubic resize to 256,
center crop to 224, scaled to [-1, 1]."""

from pathlib import Path
import io

import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision import transforms

imagenet = Path(__file__).resolve().parents[2] / "datasets/imagenet-256/data"
preprocess = transforms.Compose(
    [
        transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def load(pattern, every, count):
    """The first `count` images of every `every`-th row group: the files are
    sorted by class, so this spreads the images over the classes."""
    images, labels = [], []
    for path in sorted(imagenet.glob(pattern)):
        data = pq.ParquetFile(path)
        for group in range(0, data.num_row_groups, every):
            for row in data.read_row_group(group).slice(0, count).to_pylist():
                image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                images.append(preprocess(image))
                labels.append(row["label"])
    return torch.stack(images), torch.tensor(labels)

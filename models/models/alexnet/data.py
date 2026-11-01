"""ImageNet-256 images as the checkpoint takes them: resized to 256, center
cropped to 224 and normalized with the ImageNet mean and deviation."""

from pathlib import Path
import io

import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision import transforms

imagenet = Path(__file__).resolve().parents[2] / "datasets/imagenet-256/data"
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def load(pattern, every):
    """The first image of every `every`-th row group: the files are sorted by
    class, so this spreads the images over the classes."""
    images, labels = [], []
    for path in sorted(imagenet.glob(pattern)):
        data = pq.ParquetFile(path)
        for group in range(0, data.num_row_groups, every):
            row = data.read_row_group(group).slice(0, 1).to_pylist()[0]
            image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            images.append(preprocess(image))
            labels.append(row["label"])
    return torch.stack(images), torch.tensor(labels)

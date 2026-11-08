"""ImageNet-256 images as the checkpoint takes them: resized to 224 x 224 and
normalized with the ImageNet mean and deviation."""

from pathlib import Path
import io

import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from PIL import Image
from torchvision import transforms

imagenet = Path(__file__).resolve().parents[2] / "datasets/imagenet-256/data"
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
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


def shuffled(split, count):
    """`count` images of `split` drawn by a shuffle with seed 0, in batches of
    64."""
    files = sorted(str(path) for path in imagenet.glob(f"{split}-*.parquet"))
    samples = load_dataset("parquet", data_files=files, split="train")
    samples = samples.shuffle(seed=0).select(range(count))
    for start in range(0, count, 64):
        batch = samples[start : start + 64]
        yield (
            torch.stack([preprocess(image.convert("RGB")) for image in batch["image"]]),
            torch.tensor(batch["label"]),
        )

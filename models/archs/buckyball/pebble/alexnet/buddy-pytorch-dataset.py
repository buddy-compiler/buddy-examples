# ===- buddy-pytorch-dataset.py --------------------------------------------
#
# AlexNet on pebble: the dataset of the accuracy runs (dataset.cpp), the
# 50000 ImageNet validation images, in <output>/dataset/:
#   pixels.data     each image resized to 256, center cropped to 224, as
#                   224x224x3 uint8 (HWC); dataset.cpp normalizes it as the
#                   baremetal image is
#   reference.data  the 1000 logits of the fixed-point model the device runs
#                   on each normalized image
#   labels.data     each label, int64
# Images and logits are computed by 32 processes of 2 threads: many threads
# on one process are slower on a loaded host.
#
#   buddy-pytorch-dataset.py <output>
#
# ===---------------------------------------------------------------------===

from importlib import import_module
from pathlib import Path
import io
import json
import multiprocessing
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision import transforms

from frontend.quant.graph import smooth_classifier
from models.archs.buckyball.pebble.fixed_reference import FixedGraph

model_dir = Path(__file__).resolve().parent
output = Path(sys.argv[1]) / "dataset"
output.mkdir(parents=True, exist_ok=True)
imagenet = Path(import_module("models.models.alexnet.data").imagenet)
files = sorted(imagenet.glob("val-*.parquet"))
rows = [
    (path, group, row)
    for path in files
    for group in range(pq.ParquetFile(path).num_row_groups)
    for row in range(pq.ParquetFile(path).metadata.row_group(group).num_rows)
]
samples = len(rows)
crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
deviation = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def reference():
    """The fixed-point model of buddy-pytorch-importer.py: the classifier
    smoothed, the calibrated scales."""
    model = import_module("models.models.alexnet.model").load()
    calibration = json.loads((model_dir / "output/calibration.json").read_text())
    smooth_classifier(model, calibration["smooth_classifier"])
    return FixedGraph(model, calibration)


def work(part):
    """Images first + part, first + part + parts, ...: their pixels, logits
    and labels written in place."""
    torch.set_num_threads(2)
    graph = reference()
    pixels = np.memmap(
        output / "pixels.data", np.uint8, "r+", shape=(samples, 224, 224, 3)
    )
    logits = np.memmap(
        output / "reference.data", np.float32, "r+", shape=(samples, 1000)
    )
    labels = np.memmap(output / "labels.data", np.int64, "r+", shape=(samples,))
    tables = {}
    for index in range(part, samples, parts):
        path, group, row = rows[index]
        if (path, group) not in tables:
            tables = {(path, group): pq.ParquetFile(path).read_row_group(group)}
        record = tables[path, group].slice(row, 1).to_pylist()[0]
        image = crop(Image.open(io.BytesIO(record["image"]["bytes"])).convert("RGB"))
        pixels[index] = np.asarray(image, dtype=np.uint8)
        labels[index] = record["label"]
        # The normalization of dataset.cpp, in its order.
        values = pixels[index].astype(np.float32) / np.float32(255)
        values = (values - mean) / deviation
        with torch.no_grad():
            logits[index] = graph.run(
                torch.from_numpy(values.transpose(2, 0, 1).copy())[None]
            )[0].numpy()
    for array in (pixels, logits, labels):
        array.flush()


parts = 32
for name, dtype, shape in (
    ("pixels.data", np.uint8, (samples, 224, 224, 3)),
    ("reference.data", np.float32, (samples, 1000)),
    ("labels.data", np.int64, (samples,)),
):
    np.memmap(output / name, dtype, "w+", shape=shape).flush()
# Forked: the workers take the rows and paths above as they are.
with multiprocessing.get_context("fork").Pool(parts) as pool:
    pool.map(work, range(parts))
print(f"{samples} ImageNet validation images in {output}")

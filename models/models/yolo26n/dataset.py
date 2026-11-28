"""Prepare the 64 COCO128 evaluation images and their letterboxed annotations."""

from importlib import import_module
from pathlib import Path
import json
import shutil
import sys

import numpy as np
import torch

output, quant_reference = map(Path, sys.argv[1:3])
layout = sys.argv[3]
data = output / "dataset"
evaluation = output / "detection"
data.mkdir(exist_ok=True, parents=True)
evaluation.mkdir(exist_ok=True)
labels_file = data / "labels.data"
labels_file.unlink(missing_ok=True)
coco = import_module("models.models.yolo26n.data")
assert len(coco.paths) == 128
paths = coco.paths[64:]
pixels, boxes = coco.load(paths)
ids = np.asarray([int(p.stem) for p in paths], dtype="<i8")
assert len(set(ids)) == 64
images = {"NHWC": pixels.permute(0, 2, 3, 1), "NCHW": pixels}[layout]
images.contiguous().numpy().astype("<f4", copy=False).tofile(data / "pixels.data")
assert quant_reference.stat().st_size == len(paths) * 300 * 6 * 4
shutil.copyfile(quant_reference, data / "reference.data")
torch.set_num_threads(4)
model = import_module("models.models.yolo26n.model").load().eval()
with torch.inference_mode(), (evaluation / "fp32.data").open("wb") as stream:
    for image in pixels.split(1):
        model(image).reshape(300, 6).float().numpy().astype("<f4", copy=False).tofile(
            stream
        )
(evaluation / "ground-truth.json").write_text(
    json.dumps({str(i): b.tolist() for i, b in zip(ids, boxes, strict=True)}) + "\n"
)
ids.tofile(data / "labels.tmp")
(data / "labels.tmp").replace(labels_file)
print(f"Prepared {len(ids)} held-out COCO128 images in {layout} order")

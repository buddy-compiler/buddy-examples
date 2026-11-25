# ===- buddy-pytorch-dataset.py --------------------------------------------
#
# MLP-Mixer on goban: the dataset of the accuracy runs (dataset.cpp), every
# twentieth ImageNet validation image, 2500 over all the classes (the files are
# sorted by class; the accuracy run of the design stays within two hours), in
# <output>/dataset/:
#   pixels.data     each image resized (bicubic, short side 256) and center
#                   cropped to 224, as 224x224x3 uint8 (HWC); dataset.cpp
#                   scales and patches it as the baremetal sample is
#   reference.data  the 1000 logits of the INT8 reference of
#                   buddy-pytorch-importer.py on each
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

from compiler.frontend.quant import quantize as quantize_codes, quantize_bias
from frontend.quant.graph import smooth_norms
from models.archs.buckyball.goban.exp import gelu

model_dir = Path(__file__).resolve().parent
output = Path(sys.argv[1]) / "dataset"
output.mkdir(parents=True, exist_ok=True)
mixer = import_module("models.models.mlp-mixer.model")
calibration = json.loads((model_dir / "output/calibration.json").read_text())
imagenet = Path(import_module("models.models.alexnet.data").imagenet)
files = sorted(imagenet.glob("val-*.parquet"))
rows = [
    (path, group, row)
    for path in files
    for group in range(pq.ParquetFile(path).num_row_groups)
    for row in range(pq.ParquetFile(path).metadata.row_group(group).num_rows)
][::20]
samples = len(rows)


def reference():
    """The INT8 model of the baremetal check of buddy-pytorch-importer.py."""
    model = mixer.load()
    smooth_norms(model, calibration["smooth"])
    for name, module in model.named_modules():
        # GELU as the goban operators compute it (exp.py).
        if isinstance(module, torch.nn.GELU):
            module.forward = gelu
        if not isinstance(module, torch.nn.Linear):
            continue
        step = module.weight.detach().abs().amax(1, keepdim=True) / 127
        module.weight.data = quantize_codes(module.weight.detach(), 1 / step) * step
        if "mlp_tokens" in name:
            scale = torch.tensor(calibration[name]).reshape(-1, 1) / 127
        else:
            scale = torch.full([1, 1], calibration[name] / 127)
            bias_step = step.reshape(-1) * scale.reshape(-1)
            module.bias.data = (
                quantize_bias(module.bias.detach(), bias_step) * bias_step
            )
        module.register_forward_pre_hook(
            lambda module, args, scale=scale: quantize_codes(args[0], 1 / scale) * scale
        )
    return model


def work(part):
    """Images part, part + parts, ...: their pixels, logits and labels written
    in place."""
    torch.set_num_threads(2)
    model = reference()
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
        image = Image.open(io.BytesIO(record["image"]["bytes"])).convert("RGB")
        width, height = image.size
        size = (
            (256, 256 * height // width)
            if width <= height
            else (256 * width // height, 256)
        )
        image = image.resize(size, Image.Resampling.BICUBIC)
        left, top = round((image.width - 224) / 2), round((image.height - 224) / 2)
        pixels[index] = np.asarray(
            image.crop((left, top, left + 224, top + 224)), dtype=np.uint8
        )
        labels[index] = record["label"]
        # The scaling and patches of dataset.cpp, in its order.
        values = pixels[index].astype(np.float32) / np.float32(127.5) - np.float32(1.0)
        patches = (
            values.transpose(2, 0, 1)
            .reshape(3, 14, 16, 14, 16)
            .transpose(1, 3, 0, 2, 4)
            .reshape(196, 768)
        )
        with torch.no_grad():
            logits[index] = model(
                torch.from_numpy(np.ascontiguousarray(patches))[None]
            )[0, 0].numpy()
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

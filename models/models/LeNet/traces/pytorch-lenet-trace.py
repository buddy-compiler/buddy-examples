#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import LeNet


CHECKPOINTS = [
    ("input_nchw", "nchw"),
    ("conv1_out_nchw", "nchw"),
    ("relu1_out_nchw", "nchw"),
    ("pool1_out_nchw", "nchw"),
    ("conv2_out_nchw", "nchw"),
    ("relu2_out_nchw", "nchw"),
    ("pool2_out_nchw", "nchw"),
    ("flatten_out", "nc"),
    ("fc1_out", "nc"),
    ("relu3_out", "nc"),
    ("fc2_out", "nc"),
    ("relu4_out", "nc"),
    ("fc3_out", "nc"),
]


def write_trace(trace_path: Path, records: list[dict]) -> None:
    with trace_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def record(tag: str, layout: str, tensor: torch.Tensor) -> dict:
    array = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
    return {
        "tag": tag,
        "layout": layout,
        "shape": list(array.shape),
        "values": array.reshape(-1).tolist(),
    }


def load_input(args: argparse.Namespace) -> torch.Tensor:
    if args.input_bin:
        data = np.fromfile(args.input_bin, dtype=np.float32)
        if data.size != 28 * 28:
            raise ValueError(f"expected 784 floats in {args.input_bin}, got {data.size}")
        return torch.from_numpy(data.reshape(1, 1, 28, 28)).to(torch.float32)

    image = Image.open(args.image).convert("L")
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return transform(image).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image")
    parser.add_argument("--input-bin")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model = torch.load(model_dir / "lenet-model.pth", weights_only=False)
    model.eval()

    x = load_input(args)
    traces = [record("input_nchw", "nchw", x)]

    with torch.no_grad():
      conv1 = model.conv1(x)
      traces.append(record("conv1_out_nchw", "nchw", conv1))
      relu1 = torch.relu(conv1)
      traces.append(record("relu1_out_nchw", "nchw", relu1))
      pool1 = model.pool(relu1)
      traces.append(record("pool1_out_nchw", "nchw", pool1))

      conv2 = model.conv2(pool1)
      traces.append(record("conv2_out_nchw", "nchw", conv2))
      relu2 = torch.relu(conv2)
      traces.append(record("relu2_out_nchw", "nchw", relu2))
      pool2 = model.pool(relu2)
      traces.append(record("pool2_out_nchw", "nchw", pool2))

      flatten = pool2.view(-1, 16 * 4 * 4)
      traces.append(record("flatten_out", "nc", flatten))
      fc1 = model.fc1(flatten)
      traces.append(record("fc1_out", "nc", fc1))
      relu3 = torch.relu(fc1)
      traces.append(record("relu3_out", "nc", relu3))
      fc2 = model.fc2(relu3)
      traces.append(record("fc2_out", "nc", fc2))
      relu4 = torch.relu(fc2)
      traces.append(record("relu4_out", "nc", relu4))
      fc3 = model.fc3(relu4)
      traces.append(record("fc3_out", "nc", fc3))

    write_trace(Path(args.output), traces)


if __name__ == "__main__":
    main()

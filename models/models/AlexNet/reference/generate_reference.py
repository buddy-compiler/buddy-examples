#!/usr/bin/env python3
"""Generate the canonical full-output reference for the AlexNet workload.

Model: https://huggingface.co/Pie33000/alexnet (from-scratch AlexNet,
ImageNet-1K). Original implementation: https://github.com/pie33000/alexnet
(`model.py`, `load_data.py`). Checkpoint: model_40.pth (epoch 40).

Pipeline (single crop, same as the author's validation pre-processing with
TenCrop replaced by CenterCrop):
  images/dog-326x256.bmp
    -> Resize(256)    [identity: short side is already 256]
    -> CenterCrop(224)
    -> ToTensor()     [x / 255]
    -> Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    -> model          [eval mode; dropout disabled]
    -> logits (1, 1000)

Outputs:
  reference/alexnet-reference.npz  - input tensor, full logits, full softmax
  reference/alexnet-reference.txt  - human-readable dump (logits + probs +
                                     top-5 with labels)

Usage: python3 reference/generate_reference.py [--checkpoint PATH]
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

HERE = Path(__file__).resolve().parent          # .../AlexNet/reference
MODEL_DIR = HERE.parent                          # .../AlexNet
LABELS = MODEL_DIR / "Labels.txt"
ASSET = MODEL_DIR / "images" / "dog-326x256.bmp"
OUT_NPZ = HERE / "alexnet-reference.npz"
OUT_TXT = HERE / "alexnet-reference.txt"

CKPT_URL = "https://huggingface.co/Pie33000/alexnet/resolve/main/model_40.pth"
CKPT_DEFAULT = MODEL_DIR / "model_40.pth"


class AlexNet(torch.nn.Module):
    """Architecture identical to https://github.com/pie33000/alexnet/model.py."""

    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, kernel_size=(11, 11), stride=4, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(96, 256, kernel_size=(5, 5), stride=1, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(256, 384, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 384, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(256 * 13 * 13, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CKPT_DEFAULT))
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"downloading checkpoint {CKPT_URL} -> {ckpt}")
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CKPT_URL, ckpt)

    model = AlexNet()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()

    tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img = Image.open(ASSET).convert("RGB")
    x = tf(img).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=1)
    logits_np = logits.numpy()[0]
    probs_np = probs.numpy()[0]

    labels = [line.strip() for line in open(LABELS) if line.strip()]
    top5 = np.argsort(probs_np)[::-1][:5]

    np.savez(OUT_NPZ, input=x.numpy(), logits=logits_np, probs=probs_np)
    with open(OUT_TXT, "w") as f:
        f.write("AlexNet reference output (Pie33000/alexnet, model_40.pth)\n")
        f.write(f"input asset: {ASSET.name}\n")
        f.write(f"input: {tuple(x.shape)} min={float(x.min()):.6f} max={float(x.max()):.6f}\n")
        f.write("logits:\n")
        for v in logits_np:
            f.write(f"{v:.8e}\n")
        f.write("probs:\n")
        for v in probs_np:
            f.write(f"{v:.8e}\n")
        f.write("top5:\n")
        for rank, idx in enumerate(top5):
            f.write(f"{rank + 1},{idx},{labels[idx]},{probs_np[idx]:.8e},{logits_np[idx]:.8e}\n")

    print("top-5:")
    for rank, idx in enumerate(top5):
        print(f"  {rank + 1}: [{idx}] {labels[idx]:40s} p={probs_np[idx]:.6f}")
    print(f"wrote {OUT_NPZ}")
    print(f"wrote {OUT_TXT}")


if __name__ == "__main__":
    main()

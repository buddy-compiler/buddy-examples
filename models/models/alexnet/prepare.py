from pathlib import Path
import tomllib

import numpy as np
import torch
from PIL import Image
from torchvision.models import AlexNet_Weights, alexnet


with Path(__file__).with_name("configs").joinpath("model.toml").open("rb") as file:
    WEIGHTS = tomllib.load(file)["parameters"]["weights"]


def load_model(weights=WEIGHTS):
    model = alexnet(weights=AlexNet_Weights[weights]).eval()
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False
    return model


def load_input(path: Path):
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    if pixels.shape != (224, 224, 3):
        raise ValueError(f"AlexNet input must be 224x224 RGB, got {pixels.shape}")
    value = pixels / np.float32(255.0)
    value = (value - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(value.transpose(2, 0, 1).copy())[None]


if __name__ == "__main__":
    torch.set_num_threads(4)
    model = load_model()
    with torch.no_grad():
        logits = model(load_input(Path(__file__).parent / "images/dog.bmp"))
    print("PyTorch top5:", logits.topk(5).indices.tolist())

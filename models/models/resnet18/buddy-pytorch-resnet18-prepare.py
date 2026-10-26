from pathlib import Path
import torch
from torchvision.models import resnet18, ResNet18_Weights

output = Path(__file__).resolve().parent / "output"
(output / "resnet18").mkdir(parents=True, exist_ok=True)
model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
torch.save(model.state_dict(), output / "resnet18/model.pth")

from huggingface_hub import snapshot_download

snapshot_download(
    "evanarlian/imagenet_1k_resized_256",
    repo_type="dataset",
    allow_patterns=["data/train-*.parquet", "data/val-*.parquet"],
    local_dir=output.parents[2] / "datasets/imagenet-256",
)

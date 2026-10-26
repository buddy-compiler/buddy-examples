from pathlib import Path
import torch
from torchvision.models import alexnet, AlexNet_Weights
from huggingface_hub import snapshot_download

output = Path(__file__).resolve().parent / "output"
(output / "alexnet").mkdir(parents=True, exist_ok=True)
model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).eval()
torch.save(model.state_dict(), output / "alexnet/model.pth")

# ImageNet-1K resized to 256 pixels: train for calibration, validation for evaluation.
snapshot_download(
    "evanarlian/imagenet_1k_resized_256",
    repo_type="dataset",
    allow_patterns=["data/train-*.parquet", "data/val-*.parquet"],
    local_dir=output.parents[2] / "datasets/imagenet-256",
)

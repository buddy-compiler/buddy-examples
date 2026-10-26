from pathlib import Path
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

output = Path(__file__).resolve().parent / "output"
(output / "mobilenet-v3-small").mkdir(parents=True, exist_ok=True)
model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()
torch.save(model.state_dict(), output / "mobilenet-v3-small/model.pth")

from huggingface_hub import snapshot_download

snapshot_download(
    "evanarlian/imagenet_1k_resized_256",
    repo_type="dataset",
    allow_patterns=["data/train-*.parquet", "data/val-*.parquet"],
    local_dir=output.parents[2] / "datasets/imagenet-256",
)

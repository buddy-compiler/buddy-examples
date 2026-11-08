from pathlib import Path
import os
from ultralytics import YOLO

output = Path(__file__).resolve().parent / "output/yolo26n"
output.mkdir(parents=True, exist_ok=True)
os.chdir(output)
YOLO("yolo26n.pt")

from torch.hub import download_url_to_file
from zipfile import ZipFile

datasets = Path(__file__).resolve().parents[2] / "datasets"
datasets.mkdir(parents=True, exist_ok=True)
archive = datasets / "coco128.zip"
download_url_to_file(
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
    str(archive),
)
with ZipFile(archive) as data:
    data.extractall(datasets)

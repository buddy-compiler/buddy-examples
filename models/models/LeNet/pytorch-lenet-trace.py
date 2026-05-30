#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import numpy as np
import torch
import torch._inductor.lowering
from PIL import Image
from torch._inductor.decomposition import decompositions as inductor_decomp
from torchvision import transforms

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import LeNet


def load_trace_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"trace config not found: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    items = data.get("trace")
    if not isinstance(items, list):
        raise ValueError("trace config must contain [[trace]] entries")

    result = {}
    ids = set()
    for item in items:
        node = item.get("node")
        trace_id = item.get("id")
        tag = item.get("tag")
        layout = item.get("layout")
        if not isinstance(node, str) or not node:
            raise ValueError("trace.node must be a non-empty string")
        if not isinstance(trace_id, int):
            raise ValueError(f"trace id for {node} must be an integer")
        if not isinstance(tag, str) or not tag:
            raise ValueError(f"trace tag for {node} must be a non-empty string")
        if not isinstance(layout, str) or not layout:
            raise ValueError(f"trace layout for {node} must be a non-empty string")
        if node in result:
            raise ValueError(f"duplicate trace node: {node}")
        if trace_id in ids:
            raise ValueError(f"duplicate trace id: {trace_id}")
        ids.add(trace_id)
        result[node] = {"id": trace_id, "tag": tag, "layout": layout}
    return result


def load_input(args: argparse.Namespace) -> torch.Tensor:
    if args.input_bin and args.image:
        raise ValueError("use either --image or --input-bin, not both")
    if args.input_bin:
        data = np.fromfile(args.input_bin, dtype=np.float32)
        if data.size != 28 * 28:
            raise ValueError(f"expected 784 floats in {args.input_bin}, got {data.size}")
        return torch.from_numpy(data.reshape(1, 1, 28, 28)).to(torch.float32)
    if not args.image:
        raise ValueError("one of --image or --input-bin is required")

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
    parser.add_argument("--model-dir", default=Path(__file__).resolve().parent)
    parser.add_argument("--image")
    parser.add_argument("--input-bin")
    parser.add_argument("--output")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    trace_dir = model_dir / "trace"
    output_dir = model_dir / "output"
    output = Path(args.output) if args.output else output_dir / "pytorch-trace.jsonl"

    trace_config = load_trace_config(trace_dir / "trace.toml")
    model = torch.load(model_dir / "lenet-model.pth", weights_only=False).eval()
    data = load_input(args)

    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        trace_config=trace_config,
        trace_dump_dir=output_dir,
        trace_output_path=output,
    )
    with torch.no_grad():
        compiler.importer(model, data)


if __name__ == "__main__":
    main()

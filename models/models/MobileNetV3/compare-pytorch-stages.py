#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.ops.misc import SqueezeExcitation


def nhwc(value: torch.Tensor) -> np.ndarray:
    result = value.detach().cpu().numpy()
    if result.ndim == 4:
        result = result.transpose(0, 2, 3, 1)
    return result


def load_i8_trace(trace_dir: Path, stage: int) -> np.ndarray:
    parts = sorted(
        trace_dir.glob(f"trace-{stage}-part-*.i8"),
        key=lambda path: int(path.stem.split("-")[-1]),
    )
    numbers = [int(path.stem.split("-")[-1]) for path in parts]
    if not parts or numbers != list(range(len(parts))):
        raise ValueError(f"trace {stage} parts are missing: {numbers}")
    return np.concatenate([np.fromfile(path, dtype=np.int8) for path in parts])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MobileNetV3 BEMU stages with float PyTorch"
    )
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("mlir", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--detail-stage", type=int)
    args = parser.parse_args()

    model = mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
    ).eval()
    pixels = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32)
    if pixels.shape != (224, 224, 3):
        raise ValueError(f"expected a 224x224 RGB image, got {pixels.shape}")
    value = torch.from_numpy((pixels / np.float32(255.0)).copy()).permute(2, 0, 1)[
        None
    ]

    references: list[tuple[str, np.ndarray]] = []
    with torch.no_grad():
        value = model.features[0](value)
        references.append(("conv2d", nhwc(value)))
        for feature in model.features[1:12]:
            residual = value
            for operation in feature.block:
                if isinstance(operation, SqueezeExcitation):
                    se_input = value
                    value = operation.avgpool(value)
                    references.append(("global_avg", nhwc(value)))
                    value = operation.activation(operation.fc1(value))
                    references.append(("conv2d", nhwc(value)))
                    value = operation.scale_activation(operation.fc2(value))
                    references.append(("conv2d", nhwc(value)))
                    value = value * se_input
                    references.append(("mul", nhwc(value)))
                else:
                    value = operation(value)
                    kind = (
                        "depthwise"
                        if operation[0].groups == operation[0].in_channels
                        and operation[0].in_channels != 1
                        else "conv2d"
                    )
                    references.append((kind, nhwc(value)))
            if feature.use_res_connect:
                value = value + residual
                references.append(("add", nhwc(value)))
        value = model.features[12](value)
        references.append(("conv2d", nhwc(value)))
        value = model.avgpool(value)
        references.append(("global_avg", nhwc(value)))
        value = torch.flatten(value, 1)
        value = model.classifier[1](model.classifier[0](value))
        classifier0 = value.detach().cpu().numpy()
        logits = model.classifier[3](value).detach().cpu().numpy()

    if len(references) != 77:
        raise ValueError(f"expected 77 PyTorch stages, found {len(references)}")
    mlir = args.mlir.read_text()
    stage_lines = [
        line for line in mlir.splitlines() if "mega_kernel_size = 77 : i64" in line
    ]
    if len(stage_lines) != 77:
        raise ValueError(f"expected 77 MLIR stages, found {len(stage_lines)}")

    print("stage  op          corr       mae       max")
    for stage, ((kind, reference), line) in enumerate(zip(references, stage_lines)):
        number = re.search(r"mega_kernel_stage = (\d+) : i64", line)
        scale = re.search(r"output_scale = ([^ ,}]+)", line)
        output = re.search(r"outs\(.*?tensor<([0-9x]+)xi8>", line)
        if not number or int(number.group(1)) != stage or not scale or not output:
            raise ValueError(f"malformed MLIR stage {stage}: {line}")
        shape = tuple(int(dimension) for dimension in output.group(1).split("x"))
        if reference.shape != shape:
            raise ValueError(
                f"stage {stage} PyTorch shape {reference.shape} != MLIR {shape}"
            )
        quantized_actual = load_i8_trace(args.trace_dir, stage).reshape(shape)
        actual = quantized_actual.astype(np.float32)
        lane_scales = re.search(
            r"lane_output_scales = array<f32: ([^>]*)>", line
        )
        if lane_scales:
            scales = np.asarray(
                [np.float32(value) for value in lane_scales.group(1).split(",")],
                dtype=np.float32,
            )
            if scales.shape != (shape[-1],):
                raise ValueError(
                    f"stage {stage} has {scales.size} lane scales for "
                    f"{shape[-1]} channels"
                )
            actual *= scales
        else:
            actual *= np.float32(scale.group(1))
        actual_flat = actual.reshape(-1).astype(np.float64)
        reference_flat = reference.reshape(-1).astype(np.float64)
        corr = float(np.corrcoef(actual_flat, reference_flat)[0, 1])
        difference = np.abs(actual_flat - reference_flat)
        print(
            f"{stage:5d}  {kind:10s}  {corr: .6f}  "
            f"{np.mean(difference):8.5f}  {np.max(difference):8.5f}"
        )
        if args.detail_stage == stage:
            print("channel  corr       mae       max   saturation")
            for channel in range(shape[-1]):
                actual_channel = actual[..., channel].reshape(-1).astype(np.float64)
                reference_channel = (
                    reference[..., channel].reshape(-1).astype(np.float64)
                )
                if np.std(actual_channel) == 0.0 or np.std(reference_channel) == 0.0:
                    channel_corr = float("nan")
                else:
                    channel_corr = float(
                        np.corrcoef(actual_channel, reference_channel)[0, 1]
                    )
                channel_difference = np.abs(actual_channel - reference_channel)
                quantized_channel = quantized_actual[..., channel]
                saturation = np.mean(
                    (quantized_channel == -128) | (quantized_channel == 127)
                )
                print(
                    f"{channel:7d}  {channel_corr: .6f}  "
                    f"{np.mean(channel_difference):8.5f}  "
                    f"{np.max(channel_difference):8.5f}  {saturation:10.6f}"
                )

    classifier_line = next(
        line
        for line in mlir.splitlines()
        if "mega_kernel_size = 2 : i64" in line
        and "mega_kernel_stage = 0 : i64" in line
    )
    classifier_scale = re.search(r"output_scale = ([^ ,}]+)", classifier_line)
    if not classifier_scale:
        raise ValueError("classifier stage 0 has no output scale")
    actual = load_i8_trace(args.trace_dir, 1000).reshape(1, 1024).astype(np.float32)
    actual *= np.float32(classifier_scale.group(1))
    corr = float(np.corrcoef(actual.reshape(-1), classifier0.reshape(-1))[0, 1])
    difference = np.abs(actual.astype(np.float64) - classifier0.astype(np.float64))
    print(
        f"classifier 0 corr={corr:.6f} mae={np.mean(difference):.5f} "
        f"max={np.max(difference):.5f}"
    )

    actual_logits = np.loadtxt(args.trace_dir / "trace-1001.txt", dtype=np.float32)
    if actual_logits.size != 1000:
        raise ValueError(f"expected 1000 logits, found {actual_logits.size}")
    actual_logits = actual_logits.reshape(1, 1000)
    corr = float(np.corrcoef(actual_logits.reshape(-1), logits.reshape(-1))[0, 1])
    difference = np.abs(actual_logits.astype(np.float64) - logits.astype(np.float64))
    print(
        f"classifier 1 corr={corr:.6f} mae={np.mean(difference):.5f} "
        f"max={np.max(difference):.5f} "
        f"BEMU={actual_logits.argmax(1).item()} PyTorch={logits.argmax(1).item()}"
    )


if __name__ == "__main__":
    main()

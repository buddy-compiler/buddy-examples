"""Compute detection mAP across the complete cohort, using the recipe's matching rules."""

import json
from pathlib import Path
import re
import sys

import numpy as np
import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ap_per_class, box_iou


def mean_ap(rows, targets, validator):
    correct, scores, classes, labels = [], [], [], []
    for det, target in zip(rows, targets, strict=True):
        boxes = torch.from_numpy(det)
        target = torch.as_tensor(target, dtype=torch.float32).reshape(-1, 5)
        correct.append(
            validator.match_predictions(
                boxes[:, 5], target[:, 0], box_iou(target[:, 1:], boxes[:, :4])
            )
        )
        scores.append(boxes[:, 4])
        classes.append(boxes[:, 5])
        labels.append(target[:, 0])
    ap = ap_per_class(
        *(torch.cat(v).numpy() for v in (correct, scores, classes, labels))
    )[5]
    return float(ap.mean())


if __name__ == "__main__":
    model, logs = map(Path, sys.argv[1:])
    ids = np.fromfile(model / "dataset/labels.data", dtype="<i8").tolist()
    labels = json.loads((model / "detection/ground-truth.json").read_text())
    assert set(map(int, labels)) == set(ids) and len(ids) == len(set(ids))
    fp32 = np.fromfile(model / "detection/fp32.data", dtype="<f4").reshape(
        len(ids), 300, 6
    )
    observed = {}
    passed = 0
    for log in sorted(logs.glob("shard[0-9]*/run.log")):
        text = log.read_text()
        (counts,) = re.findall(
            r"^Detection dataset: (\d+) samples, (\d+) passed\s*$", text, re.M
        )
        count, matching = map(int, counts)
        detections = re.findall(r"^YOLO boxes (\d+) ([0-9a-f]+)\s*$", text, re.M)
        first = int(log.parent.name.removeprefix("shard"))
        if (
            count != len(detections)
            or [int(i) for i, _ in detections] != ids[first : first + count]
        ):
            raise ValueError(f"Image IDs do not match the shard at {log}")
        if not 0 <= matching <= count:
            raise ValueError(f"Invalid reference comparison count at {log}")
        for label, encoded in detections:
            image = int(label)
            if image in observed:
                raise ValueError(f"Duplicate detections for image {image}")
            rows = (
                np.frombuffer(bytes.fromhex(encoded), dtype=">f4")
                .astype(np.float32)
                .reshape(300, 6)
            )
            if not np.isfinite(rows).all():
                raise ValueError(f"Non-finite detections for image {image}")
            observed[image] = rows
        passed += matching
    if set(observed) != set(ids):
        raise ValueError(f"Expected {len(ids)} images, found {len(observed)}")
    targets = [labels[str(i)] for i in ids]
    validator = DetectionValidator(save_dir=logs)
    result = dict(
        samples=len(ids),
        ran=len(observed),
        pass_rate=passed / len(ids),
        map50_95=mean_ap([observed[i] for i in ids], targets, validator),
        reference_map50_95=mean_ap(fp32, targets, validator),
        dataset=f"COCO128 held-out ({len(ids)} images)",
    )
    if not all(np.isfinite(result[k]) for k in ("map50_95", "reference_map50_95")):
        raise ValueError("Undefined detection mAP")
    (logs / "detection-metrics.json").write_text(json.dumps(result, indent=2) + "\n")

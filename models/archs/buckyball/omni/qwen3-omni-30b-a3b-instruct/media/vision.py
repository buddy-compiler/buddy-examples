import importlib
import math
from pathlib import Path
import struct

import numpy as np
import torch

from stack.serving.system import Connection

geometry = importlib.import_module(
    "stack.models.models.qwen3-omni-30b-a3b-instruct.vision_geometry"
)


class Vision:
    def __init__(self, directory: Path, metadata: dict, timeout: float):
        (chip,) = [c for c in metadata["system"]["chips"] if c["role"] == "media"]
        tile = metadata["system"]["media"]["vision_tile"]
        self.config = metadata["vision"]["config"]
        self.capacity = max(metadata["vision"]["buckets"])
        self.connection = Connection(directory, chip["id"], tile, timeout)
        if self.connection.execute(
            struct.pack("<4Q", 0, chip["id"], tile, 2), 8
        ) != bytes(8):
            raise RuntimeError("vision tile initialization failed")

    def encode(self, pixels, grid_thw):
        config = self.config
        grid = grid_thw.detach().cpu().to(torch.int64)
        if grid.ndim != 2 or grid.shape[1] != 3 or (grid <= 0).any():
            raise ValueError(
                "vision grid must have positive temporal, height and width dimensions"
            )
        if (grid[:, 1:] % config["spatial_merge_size"]).any():
            raise ValueError("vision grid is not aligned to spatial merge size")
        width = (
            config["in_channels"]
            * config["temporal_patch_size"]
            * config["patch_size"] ** 2
        )
        count = int(grid.prod(-1).sum())
        pixels = np.asarray(pixels.detach().cpu(), dtype="<f4")
        if (
            pixels.shape != (count, width)
            or (grid[:, 1] * grid[:, 2] > self.capacity).any()
        ):
            raise ValueError("pixel values exceed compiled vision contract")
        indices, coefficients = geometry.interpolation(
            grid,
            math.isqrt(config["num_position_embeddings"]),
            config["spatial_merge_size"],
        )
        positions = geometry.positions(grid, config["spatial_merge_size"])
        head_dim = config["hidden_size"] // config["num_heads"]
        frequencies = 1.0 / (
            10000 ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2))
        )
        angles = positions[:, :, None].float() * frequencies
        angles = torch.cat((angles[:, 0], angles[:, 1]), -1)
        angles = torch.cat((angles, angles), -1)
        cosine = angles.cos().numpy().astype("<f4")
        sine = angles.sin().numpy().astype("<f4")
        indices = indices.numpy().astype("<i8")
        coefficients = coefficients.numpy().astype("<f4")
        features = 1 + len(config["deepstack_visual_indexes"])
        results = []
        offset = 0
        for temporal, height, width in grid.tolist():
            patches = height * width
            output_count = patches // config["spatial_merge_size"] ** 2
            for _ in range(temporal):
                end = offset + patches
                request = (
                    struct.pack("<4Q", 12, patches, 0, 0) + pixels[offset:end].tobytes()
                )
                request += (
                    indices[offset:end].tobytes() + coefficients[offset:end].tobytes()
                )
                request += cosine[offset:end].tobytes() + sine[offset:end].tobytes()
                request += bytes(patches * patches)
                data = self.connection.execute(
                    request, features * output_count * config["out_hidden_size"] * 4
                )
                results.append(
                    np.frombuffer(data, dtype="<f4")
                    .copy()
                    .reshape(features, output_count, config["out_hidden_size"])
                )
                offset = end
        result = np.concatenate(results, axis=1)
        if not np.isfinite(result).all():
            raise RuntimeError("non-finite vision embeddings")
        return torch.from_numpy(result)

    def close(self):
        self.connection.close()

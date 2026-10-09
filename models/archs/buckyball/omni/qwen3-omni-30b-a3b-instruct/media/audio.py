from pathlib import Path
import struct

import numpy as np
import torch

from stack.serving.system import Connection


class Audio:
    def __init__(self, directory: Path, metadata: dict, timeout: float):
        (chip,) = [c for c in metadata["system"]["chips"] if c["role"] == "media"]
        tile = metadata["system"]["media"]["audio_tile"]
        self.config = metadata["audio"]["config"]
        self.connection = Connection(directory, chip["id"], tile, timeout)
        if self.connection.execute(
            struct.pack("<4Q", 0, chip["id"], tile, 3), 8
        ) != bytes(8):
            raise RuntimeError("audio tile initialization failed")

    def encode(self, features):
        values = np.asarray(features.detach().cpu(), dtype="<f4")
        if (
            values.ndim != 2
            or values.shape[0] != self.config["num_mel_bins"]
            or not values.shape[1]
        ):
            raise ValueError("audio features must be a nonempty mel-frequency matrix")
        frames = values.shape[1]
        count = frames // 100 * 13 + (frames % 100 + 7) // 8
        request = struct.pack("<4Q", 14, frames, 0, 0) + values.T.copy().tobytes()
        data = self.connection.execute(request, count * self.config["output_dim"] * 4)
        result = (
            np.frombuffer(data, dtype="<f4")
            .copy()
            .reshape(count, self.config["output_dim"])
        )
        if not np.isfinite(result).all():
            raise RuntimeError("non-finite audio embeddings")
        return torch.from_numpy(result)

    def close(self):
        self.connection.close()

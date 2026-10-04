from pathlib import Path
import struct

import numpy as np
import torch

from stack.serving.system import Connection


class Wave:
    def __init__(self, directory: Path, metadata: dict, timeout: float):
        (chip,) = [c for c in metadata["system"]["chips"] if c["role"] == "media"]
        tile = metadata["system"]["media"]["code2wav_tile"]
        self.capacity = max(metadata["wave"]["buckets"])
        self.directory = directory
        self.connection = Connection(directory, chip["id"], tile, timeout)
        if self.connection.execute(
            struct.pack("<4Q", 0, chip["id"], tile, 5), 8
        ) != bytes(8):
            raise RuntimeError("Code2Wav tile initialization failed")

    def decode(self, codes):
        values = np.asarray(codes.detach().cpu(), dtype="<u8")
        if (
            values.ndim != 2
            or values.shape[0] != 16
            or not 0 < values.shape[1] <= self.capacity
        ):
            raise ValueError("acoustic codes exceed compiled Code2Wav contract")
        count = values.shape[1]
        np.save(self.directory / "wave-codes.npy", values)
        request = struct.pack("<4Q", 22, count, 0, 0) + values.tobytes()
        data = self.connection.execute(request, (count * 1920 - 555) * 4)
        result = np.frombuffer(data, dtype="<f4").copy()
        if not np.isfinite(result).all():
            raise RuntimeError("non-finite waveform")
        return torch.from_numpy(result)

    def close(self):
        self.connection.close()

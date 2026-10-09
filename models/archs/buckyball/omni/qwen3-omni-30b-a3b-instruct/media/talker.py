from pathlib import Path
import struct

import numpy as np
import torch

from stack.serving.system import Connection


class Talker:
    def __init__(self, directory: Path, metadata: dict, timeout: float):
        (chip,) = [c for c in metadata["system"]["chips"] if c["role"] == "media"]
        tile = metadata["system"]["media"]["talker_tile"]
        self.config = metadata["talker"]["config"]
        self.width = self.config["text_config"]["hidden_size"]
        self.prefill = metadata["talker"]["prefill_length"]
        self.capacity = metadata["talker"]["cache_length"]
        self.connection = Connection(directory, chip["id"], tile, timeout)
        if self.connection.execute(
            struct.pack("<4Q", 0, chip["id"], tile, 4), 8
        ) != bytes(8):
            raise RuntimeError("Talker tile initialization failed")

    def embedding(self, tokens, group=0):
        tokens = np.asarray(tokens, dtype="<u8").reshape(-1)
        parts = []
        for begin in range(0, tokens.size, self.prefill):
            values = tokens[begin : begin + self.prefill]
            data = self.connection.execute(
                struct.pack("<4Q", 17, values.size, group, 0) + values.tobytes(),
                values.size * self.width * 4,
            )
            parts.append(
                np.frombuffer(data, dtype="<f4").copy().reshape(-1, self.width)
            )
        return torch.from_numpy(np.concatenate(parts))

    def project(self, hidden, kind):
        values = np.asarray(hidden.detach().cpu(), dtype="<f4").reshape(
            -1, self.config["thinker_hidden_size"]
        )
        parts = []
        for begin in range(0, values.shape[0], self.prefill):
            rows = values[begin : begin + self.prefill]
            data = self.connection.execute(
                struct.pack("<4Q", 16, rows.shape[0], kind, 0) + rows.tobytes(),
                rows.shape[0] * self.width * 4,
            )
            parts.append(
                np.frombuffer(data, dtype="<f4").copy().reshape(-1, self.width)
            )
        result = torch.from_numpy(np.concatenate(parts))
        return result.reshape(*hidden.shape[:-1], self.width)

    def forward(self, hidden, positions, start, *, predictor=False):
        values = np.asarray(hidden.detach().cpu(), dtype="<f4").reshape(-1, self.width)
        positions = np.asarray(positions.detach().cpu(), dtype="<i8")
        count = values.shape[0]
        if positions.shape != (3, count) or start + count > (
            16 if predictor else self.capacity
        ):
            raise ValueError("Talker positions exceed compiled cache contract")
        block = 2 if predictor else self.prefill
        outputs = []
        for begin in range(0, count, block):
            end = min(begin + block, count)
            request = struct.pack(
                "<4Q", 19 if predictor else 18, end - begin, start + begin, 0
            )
            request += values[begin:end].tobytes() + positions[:, begin:end].tobytes()
            data = self.connection.execute(request, (end - begin) * self.width * 4)
            outputs.append(
                np.frombuffer(data, dtype="<f4").copy().reshape(-1, self.width)
            )
        result = np.concatenate(outputs)
        if not np.isfinite(result).all():
            raise RuntimeError("non-finite Talker hidden states")
        return torch.from_numpy(result)

    def logits(self, hidden, group=None):
        values = np.asarray(hidden.detach().cpu(), dtype="<f4").reshape(-1, self.width)
        vocabulary = (
            self.config["text_config"]["vocab_size"]
            if group is None
            else self.config["code_predictor_config"]["vocab_size"]
        )
        outputs = []
        for row in values:
            request = (
                struct.pack(
                    "<4Q",
                    20 if group is None else 21,
                    1,
                    0 if group is None else group,
                    0,
                )
                + row.tobytes()
            )
            data = self.connection.execute(request, vocabulary * 4)
            outputs.append(np.frombuffer(data, dtype="<f4").copy())
        result = np.stack(outputs)
        if not np.isfinite(result).all():
            raise RuntimeError("non-finite Talker logits")
        return torch.from_numpy(result)

    def codes(self, first, hidden):
        first = first.reshape(-1)
        if first.numel() != 1 or hidden.shape != (1, 1024):
            raise ValueError("code prediction requires one frame")
        embedding = self.embedding(first.numpy())
        inputs = torch.cat((hidden, embedding))
        output = self.forward(inputs, torch.arange(2).repeat(3, 1), 0, predictor=True)
        token = self.logits(output[-1:], 0).argmax(-1)
        tokens = [first.item(), token.item()]
        total = embedding.clone()
        for group in range(14):
            embedded = self.embedding(token.numpy(), group + 1)
            total += embedded
            output = self.forward(
                embedded,
                torch.full((3, 1), group + 2, dtype=torch.int64),
                group + 2,
                predictor=True,
            )
            token = self.logits(output, group + 1).argmax(-1)
            tokens.append(token.item())
        total += self.embedding(token.numpy(), 15)
        return torch.tensor(tokens, dtype=torch.int64).reshape(1, 16, 1), total.reshape(
            1, 1, self.width
        )

    def close(self):
        self.connection.close()

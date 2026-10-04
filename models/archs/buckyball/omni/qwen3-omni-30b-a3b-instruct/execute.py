from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import struct
import time

import numpy as np

from stack.serving.system import Connection


class Thinker:
    def __init__(self, directory: Path, metadata: dict, timeout: float):
        self.config = metadata["config"]
        self.prefill = metadata["prefill_length"]
        self.capacity = metadata["cache_length"]
        self.chips = [
            chip for chip in metadata["system"]["chips"] if chip["role"] == "thinker"
        ]
        self.tiles = metadata["system"]["thinker"]["tensor_tiles"]
        self.hidden = self.config["hidden_size"]
        self.top_k = self.config["num_experts_per_tok"]
        self.events = (directory / "thinker-events.jsonl").open("w")
        self.connections = {
            (chip["id"], tile): Connection(directory, chip["id"], tile, timeout)
            for chip in self.chips
            for tile in self.tiles
        }
        self.buffers = {}
        self.pool = ThreadPoolExecutor(
            max_workers=len(self.connections), thread_name_prefix="tile"
        )
        pending = [
            self.pool.submit(
                connection.execute, struct.pack("<4Q", 0, chip, tile, 1), 8
            )
            for (chip, tile), connection in self.connections.items()
        ]
        for ready in pending:
            if ready.result() != bytes(8):
                raise RuntimeError("tile initialization failed")
        for chip in self.chips:
            identity, address, capacity = struct.unpack(
                "<3Q",
                self.connections[chip["id"], self.tiles[0]].execute(
                    struct.pack("<4Q", 11, 0, 0, 0), 24
                ),
            )
            if identity != chip["id"] or capacity < self.prefill * self.hidden * 4:
                raise ValueError(
                    "device identity or pipeline DMA buffer differs from the model deployment"
                )
            self.buffers[identity] = address

    def parallel(self, chip, request, result_bytes):
        pending = [
            self.pool.submit(
                self.connections[chip, tile].execute, request, result_bytes
            )
            for tile in self.tiles
        ]
        return [future.result() for future in pending]

    def reduce(self, chip, partials, count, residual=None):
        request = struct.pack("<4Q", 5, count, int(residual is not None), 0)
        if residual is not None:
            request += residual
        request += b"".join(partials)
        return self.connections[chip, self.tiles[0]].execute(
            request, count * self.hidden * 4
        )

    def embeddings(self, token_ids):
        tokens = np.asarray(token_ids, dtype="<i8").reshape(-1)
        if (
            not 0 < tokens.size <= self.capacity
            or (tokens < 0).any()
            or (tokens >= self.config["vocab_size"]).any()
        ):
            raise ValueError("token IDs exceed the compiled embedding contract")
        if tokens.size > self.prefill:
            return b"".join(
                self.embeddings(tokens[i : i + self.prefill])
                for i in range(0, tokens.size, self.prefill)
            )
        request = struct.pack("<4Q", 1, tokens.size, 0, 0) + tokens.tobytes()
        chip = self.chips[0]["id"]
        parts = self.parallel(chip, request, tokens.size * self.hidden * 4)
        return self.reduce(chip, parts, tokens.size)

    def forward(
        self,
        token_ids,
        positions,
        start,
        inputs_embeds=None,
        deepstack=None,
        capture=(),
    ):
        positions = np.asarray(positions, dtype="<i8")
        count = positions.shape[1]
        if positions.shape != (3, count) or start < 0 or start + count > self.capacity:
            raise ValueError("invalid MRoPE positions or cache range")
        if inputs_embeds is not None:
            inputs_embeds = np.asarray(inputs_embeds, dtype="<f4")
            if inputs_embeds.shape != (count, self.hidden):
                raise ValueError("input embedding shape differs from token sequence")
        self.captures = {index: [] for index in capture}
        chunks = []
        for begin in range(0, count, self.prefill):
            end = min(begin + self.prefill, count)
            tokens = (
                None
                if token_ids is None
                else np.asarray(token_ids, dtype="<i8")[begin:end]
            )
            embeds = None if inputs_embeds is None else inputs_embeds[begin:end]
            features = None if deepstack is None else deepstack[:, begin:end]
            chunks.append(
                self.forward_chunk(
                    tokens, positions[:, begin:end], start + begin, embeds, features
                )
            )
        self.captures = {
            index: np.concatenate(values) for index, values in self.captures.items()
        }
        return np.concatenate(chunks)

    def forward_chunk(self, tokens, positions, start, inputs_embeds, deepstack):
        count = positions.shape[1]
        hidden = (
            self.embeddings(tokens)
            if inputs_embeds is None
            else inputs_embeds.tobytes()
        )
        self.input_embeddings = (
            np.frombuffer(hidden, dtype="<f4").copy().reshape(count, self.hidden)
        )
        payload_bytes = count * self.hidden * 4
        for index, chip in enumerate(self.chips):
            if index:
                source = self.chips[index - 1]["id"]
                request = struct.pack(
                    "<4Q", 9, count, chip["id"], self.buffers[chip["id"]]
                )
                if self.connections[source, self.tiles[0]].execute(request, 8) != bytes(
                    8
                ):
                    raise RuntimeError("pipeline DMA submit failed")
                hidden = self.connections[chip["id"], self.tiles[0]].execute(
                    struct.pack("<4Q", 10, count, source, 0), payload_bytes
                )
                self.events.write(
                    json.dumps(
                        {
                            "source_chip": source,
                            "destination_chip": chip["id"],
                            "bytes": payload_bytes,
                            "transport": "chip-dma",
                        }
                    )
                    + "\n"
                )
                self.events.flush()
            for layer in range(*chip["layers"]):
                if layer in self.captures:
                    self.captures[layer].append(
                        np.frombuffer(hidden, dtype="<f4")
                        .copy()
                        .reshape(count, self.hidden)
                    )
                started = time.monotonic()
                request = (
                    struct.pack("<4Q", 2, count, start, layer)
                    + hidden
                    + positions.tobytes()
                )
                parts = self.parallel(chip["id"], request, payload_bytes)
                hidden = self.reduce(chip["id"], parts, count, hidden)
                self.events.write(
                    json.dumps(
                        {
                            "chip": chip["id"],
                            "layer": layer,
                            "start": start,
                            "tokens": count,
                            "attention_host_seconds": time.monotonic() - started,
                        }
                    )
                    + "\n"
                )
                self.events.flush()
                request = struct.pack("<4Q", 3, count, 0, layer) + hidden
                routed = self.connections[chip["id"], self.tiles[0]].execute(
                    request, payload_bytes + count * self.top_k * 12
                )
                request = struct.pack("<4Q", 4, count, 0, layer) + routed
                parts = self.parallel(chip["id"], request, payload_bytes)
                hidden = self.reduce(chip["id"], parts, count, hidden)
                if deepstack is not None and layer < deepstack.shape[0]:
                    features = [np.asarray(deepstack[layer], dtype="<f4").tobytes()]
                    features.extend(bytes(payload_bytes) for _ in self.tiles[1:])
                    hidden = self.reduce(chip["id"], features, count, hidden)
                self.events.write(
                    json.dumps(
                        {
                            "chip": chip["id"],
                            "layer": layer,
                            "start": start,
                            "tokens": count,
                            "layer_host_seconds": time.monotonic() - started,
                        }
                    )
                    + "\n"
                )
                self.events.flush()
        request = struct.pack("<4Q", 7, count, 0, 0) + hidden
        hidden = self.connections[self.chips[-1]["id"], self.tiles[0]].execute(
            request, payload_bytes
        )
        return np.frombuffer(hidden, dtype="<f4").copy().reshape(count, self.hidden)

    def logits(self, hidden):
        hidden = np.asarray(hidden, dtype="<f4")
        if hidden.shape != (1, self.hidden):
            raise ValueError("output projection consumes one hidden state")
        request = struct.pack("<4Q", 6, 1, 0, 0) + hidden.tobytes()
        width = self.config["vocab_size"] // len(self.tiles)
        parts = self.parallel(self.chips[-1]["id"], request, width * 4)
        return np.concatenate([np.frombuffer(part, dtype="<f4") for part in parts])[
            None
        ]

    def close(self):
        self.pool.shutdown(wait=True)
        for connection in self.connections.values():
            connection.close()
        self.events.close()

import json
from contextlib import closing

import numpy as np
import torch
from transformers import AutoTokenizer

from stack.serving.executor import Pool


class Model:
    def __init__(self, package, args):
        self.metadata = package.metadata
        if self.metadata["version"] != 1:
            raise ValueError("unsupported compiled BERT artifact")
        self.length = self.metadata["sequence_length"]
        self.labels = self.metadata["num_labels"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.metadata["model"])
        self.pool = Pool(
            args.simulator.resolve(),
            {"directory": str(package.directory), "program": str(package.program),
             "metadata": str(package.metadata_path)},
            args.log_dir.resolve(), args.tile_indices, args.record_io,
            [() for _ in args.tile_indices], args.memory_mib, itrace=args.itrace, mtrace=args.mtrace, timeout=args.timeout,
        )

    def classify(self, text):
        tokens = self.tokenizer(text, return_token_type_ids=True)
        count = len(tokens["input_ids"])
        if count > self.length:
            raise ValueError("input exceeds the compiled BERT sequence length")
        request = np.zeros((4, self.length), dtype="<i8")
        request[0, :count] = tokens["input_ids"]
        request[1, :count] = tokens["token_type_ids"]
        request[2, :count] = tokens["attention_mask"]
        request[3] = np.arange(self.length)
        raw = self.pool.submit(request.tobytes(), self.labels * 4).result()
        logits = np.frombuffer(raw, dtype="<f4").copy()
        if not np.isfinite(logits).all():
            raise RuntimeError("compiled BERT produced non-finite logits")
        probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).tolist()
        scores = dict(zip(self.metadata["labels"], probabilities, strict=True))
        label = max(scores, key=scores.__getitem__)
        return {"text": text, "label": label, "probability": scores[label], "scores": scores}

    def close(self):
        self.pool.close()


def run(package, args):
    if not all(isinstance(text, str) for text in args.inputs):
        raise ValueError("BERT inputs must be text strings")
    with closing(Model(package, args)) as model:
        print(json.dumps([model.classify(text) for text in args.inputs]))

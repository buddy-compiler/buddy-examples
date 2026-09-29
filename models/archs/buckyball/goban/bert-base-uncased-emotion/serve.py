import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler.seqwise import pooler_for_classify
from vllm.model_executor.models.interfaces import SupportsCrossEncoding
from vllm.model_executor.models.interfaces_base import attn_type, default_pooling_type

from stack.serving.executor import Pool


@attn_type("encoder_only")
@default_pooling_type(seq_pooling_type="CLS")
class Model(nn.Module, SupportsCrossEncoding):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config
        options = vllm_config.additional_config["execution"]
        self.artifact = options["artifact"]
        self.simulator = Path(options["simulator"]).resolve()
        self.log_root = Path(options["log_dir"]).resolve()
        self.tile_indices = options["tile_indices"]
        self.record_io = options.get("record_io", False)
        self.memory_mib = options["memory_mib"]
        self.metadata = json.loads(Path(self.artifact["metadata"]).read_text())
        if self.metadata["version"] != 1 or config.hf_config.model_type != "bert":
            raise ValueError("unsupported compiled model format")
        self.length = self.metadata["sequence_length"]
        self.labels = self.metadata["num_labels"]
        if config.max_model_len > self.length or config.hf_config.num_labels != self.labels:
            raise ValueError("vLLM model dimensions do not match the compiled artifact")
        if config.dtype != torch.float32 or not config.enforce_eager:
            raise ValueError("the compiled BERT interface requires float32 and enforce_eager")
        if config.enable_prompt_embeds or config.pooler_config.get_seq_pooling_type() != "CLS":
            raise ValueError("the compiled BERT interface accepts token IDs and CLS pooling")
        if vllm_config.parallel_config.tensor_parallel_size != 1 or vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("this artifact represents one complete tile model")
        if vllm_config.cache_config.enable_prefix_caching or vllm_config.scheduler_config.enable_chunked_prefill:
            raise ValueError("compiled BERT requires complete encoder sequences")
        self.pooler = pooler_for_classify(config.pooler_config)
        self.pool = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("embeddings are computed inside the compiled BERT model")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        expected = self.metadata["parameter_sha256"]
        seen = set()
        for name, value in weights:
            if name == "bert.embeddings.position_ids":
                continue
            digest = hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
            if digest != expected[name]:
                raise ValueError(f"checkpoint differs from the compiled artifact: {name}")
            seen.add(name)
        if seen != set(expected):
            raise ValueError(f"checkpoint parameters are missing: {sorted(set(expected) - seen)}")
        self.pool = Pool(
            self.simulator, self.artifact, self.log_root,
            self.tile_indices,
            self.record_io, [() for _ in self.tile_indices], self.memory_mib,
        )
        return seen

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                intermediate_tensors=None, inputs_embeds=None,
                token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        if self.pool is None:
            raise RuntimeError("compiled weights have not been validated")
        if inputs_embeds is not None or intermediate_tensors is not None:
            raise ValueError("compiled BERT accepts complete token sequences")
        ids = input_ids.cpu().numpy()
        pos = positions.cpu().numpy()
        starts = np.flatnonzero(pos == 0).tolist()
        if not starts or starts[0] != 0:
            raise ValueError("encoder sequence must begin at position zero")
        types = np.zeros_like(ids) if token_type_ids is None else token_type_ids.cpu().numpy()
        result = torch.empty((len(ids), self.labels), dtype=torch.float32)
        pending = []
        for start, end in zip(starts, starts[1:] + [len(ids)]):
            size = end - start
            if size > self.length or not np.array_equal(pos[start:end], np.arange(size)):
                raise ValueError("input does not match a complete compiled sequence")
            request = np.zeros((4, self.length), dtype="<i8")
            request[0, :size] = ids[start:end]
            request[1, :size] = types[start:end]
            request[2, :size] = 1
            request[3] = np.arange(self.length)
            pending.append((self.pool.submit(request.tobytes(), self.labels * 4), start, end))
        for future, start, end in pending:
            data = future.result()
            logits = np.frombuffer(data, dtype="<f4").copy()
            if not np.isfinite(logits).all():
                raise RuntimeError("compiled BERT produced non-finite logits")
            result[start:end] = torch.from_numpy(logits)
        return result


import os


def run(package, args):
    os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = str(args.timeout)
    os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "nobind"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    from vllm import LLM

    metadata = package.metadata
    model = LLM(
        model=metadata["model"],
        runner="pooling",
        dtype="float32",
        enforce_eager=True,
        max_model_len=metadata["sequence_length"],
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.memory_utilization,
        enable_prefix_caching=False,
        hf_overrides={"architectures": [f"{__package__}.serve"]},
        additional_config={"execution": {
            "artifact": {"directory": str(package.directory), "program": str(package.program),
                         "metadata": str(package.metadata_path)},
            "simulator": str(args.simulator.resolve()),
            "log_dir": str(args.log_dir.resolve()),
            "tile_indices": args.tile_indices,
            "record_io": args.record_io,
            "memory_mib": args.memory_mib,
        }},
    )
    try:
        outputs = model.classify(args.inputs)
        results = []
        for text, output in zip(args.inputs, outputs, strict=True):
            probabilities = output.outputs.probs
            scores = dict(zip(metadata["labels"], probabilities, strict=True))
            label = max(scores, key=scores.__getitem__)
            results.append({"text": text, "label": label,
                            "probability": scores[label], "scores": scores})
        print(json.dumps(results))
    finally:
        model.llm_engine.engine_core.shutdown()

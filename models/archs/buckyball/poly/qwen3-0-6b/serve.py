import hashlib
import json
import struct
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend

from stack.serving.executor import Pool


class Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config
        hf = config.hf_config
        options = vllm_config.additional_config["execution"]
        self.artifact = options["artifact"]
        self.metadata = json.loads(Path(self.artifact["metadata"]).read_text())
        if (self.metadata["version"] != 3 or self.metadata["weight_format"] != "mxfp8_e4m3" or hf.model_type != "qwen3"
                or self.metadata["model_type"] != hf.model_type):
            raise ValueError("unsupported compiled Qwen artifact")
        for key in ("hidden_size", "vocab_size"):
            if self.metadata[key] != getattr(hf, key):
                raise ValueError(f"compiled model disagrees with checkpoint: {key}")
        self.parts = self.metadata["parts"]
        self.execution_tiles = self.metadata["execution_tiles"]
        if len(options["tile_indices"]) != self.execution_tiles:
            raise ValueError("selected tile count differs from the compiled execution plan")
        self.ffn_channels = self.metadata["ffn"]["intermediate"]
        head_dim = hf.head_dim
        if head_dim != self.metadata["head_dim"]:
            raise ValueError("compiled attention head dimension differs from checkpoint")
        self.layers = self.metadata["num_layers"]
        self.heads = self.metadata["num_kv_heads"]
        self.head_size = self.metadata["head_dim"]
        self.hidden_size = self.metadata["hidden_size"]
        self.vocabulary = self.metadata["vocab_size"]
        if (hf.num_hidden_layers != self.layers or hf.num_key_value_heads != self.heads
                or hf.num_attention_heads != self.metadata["num_attention_heads"]):
            raise ValueError("compiled Qwen attention topology differs from checkpoint")
        if config.max_model_len > self.metadata["cache_length"]:
            raise ValueError("requested context exceeds the compiled capacity")
        if config.dtype != torch.float32 or not config.enforce_eager or config.enable_prompt_embeds:
            raise ValueError("compiled Qwen requires eager float32 token-ID execution")
        if vllm_config.parallel_config.tensor_parallel_size != 1 or vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("Buckyball serving coordinates the compiled tile shards; vLLM TP/PP must be one")
        if vllm_config.cache_config.enable_prefix_caching or vllm_config.scheduler_config.enable_chunked_prefill:
            raise ValueError("compiled Qwen requires complete prefill requests")
        if vllm_config.speculative_config is not None or vllm_config.cache_config.cache_dtype != "auto":
            raise ValueError("compiled Qwen uses one-token decode and float32 KV cache")
        self.cache_names = [f"{prefix + '.' if prefix else ''}model.layers.{layer}.self_attn.attn"
                            for layer in range(self.layers)]
        self.cache_layers = nn.ModuleList([
            Attention(hf.num_attention_heads, self.head_size, self.head_size ** -0.5,
                      num_kv_heads=self.heads, cache_config=vllm_config.cache_config,
                      prefix=name, attn_backend=CPUAttentionBackend)
            for name in self.cache_names
        ])
        self.options = options
        self.pool = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("the compiled embedding subgraph executes on the tile")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        expected = self.metadata["parameter_sha256"]
        seen = set()
        for name, value in weights:
            name = self.metadata["parameter_aliases"].get(name, name)
            digest = hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
            if digest != expected[name]:
                raise ValueError(f"checkpoint differs from the compiled artifact: {name}")
            seen.add(name)
        if seen != set(expected):
            raise ValueError(f"missing checkpoint parameters: {sorted(set(expected) - seen)}")
        self.pool = Pool(
            Path(self.options["simulator"]).resolve(),
            self.artifact,
            Path(self.options["log_dir"]).resolve(), self.options["tile_indices"],
            self.options.get("record_io", False), [(str(rank),) for rank in range(self.execution_tiles)],
            self.options["memory_mib"], p2e=self.options.get("p2e"),
        )
        return seen

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                intermediate_tensors=None, inputs_embeds=None) -> torch.Tensor:
        if self.pool is None:
            raise RuntimeError("compiled weights have not been validated")
        if inputs_embeds is not None or intermediate_tensors is not None:
            raise ValueError("compiled Qwen accepts token IDs")
        context = get_forward_context()
        metadata = context.attn_metadata
        first = metadata[self.cache_names[0]]
        boundaries = first.query_start_loc.tolist()
        result = torch.zeros((input_ids.numel(), self.hidden_size), dtype=torch.float32)
        local_heads = self.heads // self.execution_tiles
        context_width = self.metadata["num_attention_heads"] // self.execution_tiles * self.head_size
        for sequence, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
            count = end - begin
            start = int(positions[begin])
            if not torch.equal(positions[begin:end].cpu(), torch.arange(start, start + count)):
                raise ValueError("non-contiguous positions in a Qwen request")
            if (start + count > self.metadata["cache_length"]
                    or (start == 0 and count > self.metadata["prefill_length"])
                    or (start and count != 1)):
                raise ValueError("request does not fit the compiled Qwen phase")
            hidden_bytes = count * self.hidden_size * 4
            request = struct.pack("<QQQQ", 0, count, start, 0)
            request += input_ids[begin:end].cpu().numpy().astype("<i8").tobytes()
            data = self.pool.submit_to(0, request, hidden_bytes).result()
            hidden = np.frombuffer(data, dtype="<f4").copy().reshape(count, self.hidden_size)
            for layer_index, (name, layer) in enumerate(zip(self.cache_names, self.cache_layers)):
                blocks, heads, block_size, _ = layer.kv_cache.shape
                key, value = layer.kv_cache.view(blocks, heads, block_size * 2, self.head_size).chunk(2, dim=2)
                info = metadata[name]
                old_positions = torch.arange(start)
                old_blocks = info.block_table[sequence, old_positions // block_size].long()
                old_offsets = old_positions % block_size
                slots = info.slot_mapping[begin:end]
                if torch.any(slots < 0):
                    raise ValueError("request has an unmapped KV cache slot")
                block_ids, offsets = slots // block_size, slots % block_size
                pending = []
                for rank in range(self.execution_tiles):
                    lo = ((rank % self.parts) * 2 + rank // self.parts) * local_heads
                    hi = lo + local_heads
                    request = bytearray(struct.pack("<QQQQ", 6, count, start, layer_index))
                    request.extend(hidden.tobytes())
                    for cache in (key, value):
                        request.extend(cache[old_blocks, lo:hi, old_offsets].permute(1, 0, 2).contiguous().numpy().tobytes())
                    size = count * context_width * 4 + 2 * local_heads * count * self.head_size * 4
                    pending.append(self.pool.submit_to(rank, bytes(request), size))
                contexts = []
                for rank, future in enumerate(pending):
                    data = future.result()
                    values = np.frombuffer(data, dtype="<f4").copy()
                    if not np.isfinite(values).all():
                        raise RuntimeError("compiled attention shard produced non-finite values")
                    contexts.append(values[:count * context_width].reshape(count, context_width))
                    updated = values[count * context_width:].reshape(2, local_heads, count, self.head_size)
                    lo = ((rank % self.parts) * 2 + rank // self.parts) * local_heads
                    hi = lo + local_heads
                    key[block_ids, lo:hi, offsets] = torch.from_numpy(updated[0]).permute(1, 0, 2)
                    value[block_ids, lo:hi, offsets] = torch.from_numpy(updated[1]).permute(1, 0, 2)
                pending = []
                for rank in range(self.parts):
                    context_values = np.concatenate((contexts[rank], contexts[rank + self.parts]), axis=1)
                    request = struct.pack("<QQQQ", 7, count, start, layer_index) + context_values.tobytes()
                    pending.append((self.pool.submit_to(rank, request, hidden_bytes // 2),
                                    self.pool.submit_to(rank + self.parts, request, hidden_bytes // 2)))
                reduced = np.zeros_like(hidden)
                for first, second in pending:
                    partial = np.concatenate((np.frombuffer(first.result(), dtype="<f4").reshape(count, self.hidden_size // 2),
                                              np.frombuffer(second.result(), dtype="<f4").reshape(count, self.hidden_size // 2)), axis=1)
                    if not np.isfinite(partial).all():
                        raise RuntimeError("compiled attention projection produced non-finite values")
                    reduced += partial
                hidden += reduced
                request = struct.pack("<QQQQ", 4, count, start, layer_index) + hidden.tobytes()
                half_channels = self.ffn_channels // 2
                pending = [self.pool.submit_to(rank, request, count * half_channels * 4)
                           for rank in range(self.execution_tiles)]
                expanded = [np.frombuffer(future.result(), dtype="<f4").reshape(count, half_channels)
                            for future in pending]
                pending = []
                for rank in range(self.parts):
                    intermediate = np.concatenate((expanded[rank], expanded[rank + self.parts]), axis=1)
                    request = struct.pack("<QQQQ", 5, count, start, layer_index) + intermediate.tobytes()
                    pending.append((self.pool.submit_to(rank, request, hidden_bytes // 2),
                                    self.pool.submit_to(rank + self.parts, request, hidden_bytes // 2)))
                reduced = np.zeros_like(hidden)
                for first, second in pending:
                    partial = np.concatenate((np.frombuffer(first.result(), dtype="<f4").reshape(count, self.hidden_size // 2),
                                              np.frombuffer(second.result(), dtype="<f4").reshape(count, self.hidden_size // 2)), axis=1)
                    if not np.isfinite(partial).all():
                        raise RuntimeError("compiled FFN shard produced non-finite values")
                    reduced += partial
                hidden += reduced
            result[begin:end] = torch.from_numpy(hidden)
        return result

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pending = []
        local_vocabulary = self.vocabulary // self.execution_tiles
        for hidden in hidden_states:
            request = struct.pack("<QQQQ", 1, 1, 0, 0) + hidden.cpu().contiguous().numpy().tobytes()
            pending.append([self.pool.submit_to(rank, request, local_vocabulary * 4)
                            for rank in range(self.execution_tiles)])
        outputs = []
        for shards in pending:
            logits = np.concatenate([np.frombuffer(future.result(), dtype="<f4") for future in shards])
            if not np.isfinite(logits).all():
                raise RuntimeError("compiled Qwen produced non-finite logits")
            outputs.append(torch.from_numpy(logits))
        return torch.stack(outputs)


import os


def run_model(package, args, architecture):
    os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = str(args.timeout)
    os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "nobind"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    from vllm import LLM, SamplingParams

    metadata = package.metadata
    block_size = 32
    blocks = (metadata["cache_length"] + block_size - 1) // block_size * args.max_num_seqs + 1
    options = {
        "block_size": block_size,
        "kv_cache_memory_bytes": blocks * block_size * metadata["num_layers"] *
                                 2 * metadata["num_kv_heads"] * metadata["head_dim"] * 4,
        "enable_chunked_prefill": False,
    }
    model = LLM(
        model=metadata["model"],
        tokenizer=str(package.directory / "tokenizer"),
        runner="generate",
        dtype="float32",
        enforce_eager=True,
        max_model_len=metadata["cache_length"],
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.memory_utilization,
        enable_prefix_caching=False,
        hf_overrides={"architectures": [architecture]},
        additional_config={"execution": {
            "artifact": {"directory": str(package.directory), "program": str(package.program),
                         "metadata": str(package.metadata_path)},
            "simulator": str(args.simulator.resolve()),
            "log_dir": str(args.log_dir.resolve()),
            "tile_indices": args.tile_indices,
            "record_io": args.record_io,
            "memory_mib": args.memory_mib,
            "p2e": getattr(args, "p2e", None),
        }},
        **options,
    )
    try:
        outputs = model.generate(args.inputs, SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens))
        print(json.dumps([
            {"prompt": output.prompt, "text": output.outputs[0].text,
             "token_ids": output.outputs[0].token_ids}
            for output in outputs
        ]))
    finally:
        model.llm_engine.engine_core.shutdown()


def run(package, args):
    run_model(package, args, "examples.models.poly.qwen3-0-6b.serve")

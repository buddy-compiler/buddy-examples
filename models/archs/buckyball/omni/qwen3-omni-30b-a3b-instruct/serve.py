import json
import os
import struct

from stack.serving.system import System, Connection


def run(package, args):
    os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = str(args.timeout)
    os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "nobind"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    from vllm import LLM, SamplingParams

    metadata = package.metadata
    chips = metadata["system"]["chips"]
    media = [chip for chip in chips if chip["role"] == "media"]
    if len(media) != 1:
        raise ValueError("Omni requires one media chip")
    with System(args.simulator.resolve(), package.program, package.directory / "weights",
                args.log_dir.resolve(), args.memory_mib, args.timeout) as system:
        print(f"BEMU system: {system.directory}", flush=True)
        connections = [Connection(system.directory, media[0]["id"], tile, args.timeout)
                       for tile in metadata["system"]["media"].values()]
        for tile, connection in zip(metadata["system"]["media"].values(), connections, strict=True):
            if connection.execute(struct.pack("<4Q", 0, media[0]["id"], tile, 0), 8) != bytes(8):
                raise RuntimeError("media tile initialization failed")
        model = None
        try:
            model = LLM(model=str(package.directory / "tokenizer"), runner="generate", load_format="bemu_omni",
                dtype="float32", enforce_eager=True, max_model_len=metadata["cache_length"],
                gpu_memory_utilization=args.memory_utilization,
                max_num_seqs=args.max_num_seqs, enable_prefix_caching=False, enable_chunked_prefill=False,
                hf_overrides={"architectures": [f"{__package__}.model"]},
                additional_config={"execution": {"directory": str(system.directory),
                    "metadata": str(package.metadata_path), "timeout": args.timeout}})
            messages = [[{"role": "user", "content": [{"type": "text", "text": prompt}]}] for prompt in args.inputs]
            outputs = model.chat(messages, SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens))
            results = []
            for prompt, output in zip(args.inputs, outputs, strict=True):
                text = output.outputs[0].text
                data = text.encode()
                if connections[0].execute(struct.pack("<4Q", 8, len(data), 0, 0) + data, 8) != bytes(8):
                    raise RuntimeError("media text output failed")
                results.append({"prompt": prompt, "text": text, "token_ids": output.outputs[0].token_ids})
            (system.directory / "result.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(results), flush=True)
        finally:
            if model is not None:
                model.llm_engine.engine_core.shutdown()
            for connection in connections:
                connection.close()

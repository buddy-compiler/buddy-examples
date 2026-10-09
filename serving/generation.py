import json
import time
from contextlib import closing

import numpy as np


def run_generation(package, args, model_type):
    from transformers import AutoTokenizer

    if args.max_tokens < 1 or not np.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("generation requires positive max_tokens and nonnegative temperature")
    if not all(isinstance(prompt, str) for prompt in args.inputs):
        raise ValueError("text generation inputs must be rendered prompt strings")
    tokenizer = AutoTokenizer.from_pretrained(package.directory / "tokenizer", local_files_only=True)
    config = json.loads((package.directory / "tokenizer/generation_config.json").read_text())
    stop_ids = config["eos_token_id"]
    stop_ids = set(stop_ids if isinstance(stop_ids, list) else [stop_ids])
    rng = np.random.default_rng()
    results = []
    args.log_dir.mkdir(parents=True, exist_ok=True)
    with closing(model_type(package, args)) as model, (args.log_dir / "generation.jsonl").open("w") as trace:
        for prompt in args.inputs:
            tokens = tokenizer.encode(prompt, add_special_tokens=True)
            if not tokens or len(tokens) > package.metadata["prefill_length"]:
                raise ValueError("prompt does not fit the compiled prefill length")
            if len(tokens) + args.max_tokens - 1 > package.metadata["cache_length"]:
                raise ValueError("generation exceeds the compiled context capacity")
            cache = model.new_cache()
            generated = []
            start = 0
            request = tokens
            for step in range(args.max_tokens):
                before = time.monotonic_ns()
                logits = model.forward(request, start, cache)
                elapsed = time.monotonic_ns() - before
                if logits.ndim != 1 or not np.isfinite(logits).all():
                    raise ValueError("model must return finite one-dimensional next-token logits")
                if args.temperature == 0:
                    token = int(np.argmax(logits))
                else:
                    scores = (logits.astype(np.float64) - logits.max()) / args.temperature
                    probabilities = np.exp(scores)
                    probabilities /= probabilities.sum()
                    token = int(rng.choice(logits.size, p=probabilities))
                generated.append(token)
                trace.write(json.dumps({"request": len(results), "step": step,
                                        "phase": "prefill" if start == 0 else "decode",
                                        "start": start, "count": len(request),
                                        "host_elapsed_ns": elapsed, "token_id": token}) + "\n")
                trace.flush()
                start += len(request)
                if token in stop_ids:
                    break
                request = [token]
            results.append({"prompt": prompt, "text": tokenizer.decode(generated, skip_special_tokens=True),
                            "token_ids": generated})
    print(json.dumps(results))

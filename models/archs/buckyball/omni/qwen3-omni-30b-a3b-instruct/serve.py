import json
import os

import soundfile
from PIL import Image
from transformers import AutoProcessor

from stack.serving.system import System
from .omni_pipeline import register


def run(package, args):
    os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = str(args.timeout)
    os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "nobind"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    from vllm import SamplingParams
    from vllm_omni.entrypoints.omni import Omni

    register()
    processor = AutoProcessor.from_pretrained(
        package.directory / "tokenizer", local_files_only=True
    )
    prompts = []
    for item in args.inputs:
        if isinstance(item, str):
            content = [{"type": "text", "text": item}]
            media = {}
        else:
            if "text" not in item or set(item) - {"text", "image", "audio", "video"}:
                raise ValueError("media request requires text and known media fields")
            content, media = [], {}
            for kind, value in item.items():
                if kind == "text":
                    content.append({"type": "text", "text": value})
                    continue
                source = (args.run_config.resolve().parent / value).resolve(strict=True)
                content.append({"type": kind})
                if kind == "image":
                    media[kind] = Image.open(source).convert("RGB")
                elif kind == "audio":
                    audio, rate = soundfile.read(source, dtype="float32")
                    if audio.ndim != 1:
                        raise ValueError("audio input must be mono")
                    media[kind] = (audio, rate)
                else:
                    from vllm.assets.video import video_to_ndarrays

                    media[kind] = video_to_ndarrays(str(source))
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(
            {
                "prompt": prompt,
                "multi_modal_data": media,
                "modalities": ["text", "audio"],
            }
        )
    with System(
        args.simulator.resolve(),
        package.program,
        package.directory / "weights",
        args.log_dir.resolve(),
        args.memory_mib,
        args.timeout,
    ) as system:
        print(f"BEMU system: {system.directory}", flush=True)
        model = Omni(
            model=str(package.directory / "tokenizer"),
            deploy_config=str(package.directory / "deploy.yaml"),
            async_chunk=False,
            init_timeout=args.timeout,
            stage_init_timeout=args.timeout,
            parallel_stage_init=True,
            additional_config={
                "execution": {
                    "directory": str(system.directory),
                    "metadata": str(package.metadata_path),
                    "timeout": args.timeout,
                }
            },
        )
        try:
            parameters = [
                SamplingParams(
                    temperature=args.temperature, max_tokens=args.max_tokens
                ),
                *model.default_sampling_params_list[1:],
            ]
            results = {}
            for output in model.generate(prompts, parameters, py_generator=True):
                result = results.setdefault(
                    output.request_id, {"request_id": output.request_id}
                )
                completion = output.outputs[0]
                if output.final_output_type == "text":
                    result.update(text=completion.text, token_ids=completion.token_ids)
                    print(
                        json.dumps(
                            {"text": completion.text, "token_ids": completion.token_ids}
                        ),
                        flush=True,
                    )
                elif output.final_output_type == "audio":
                    audio = (
                        completion.multimodal_output["audio"]
                        .float()
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                    path = system.directory / f"{output.request_id}.wav"
                    soundfile.write(path, audio, 24000, subtype="FLOAT")
                    result.update(
                        audio=path.name, samples=audio.size, sample_rate=24000
                    )
                    print(
                        f"Audio complete: {audio.size} samples at 24000 Hz", flush=True
                    )
                else:
                    raise ValueError(
                        f"unexpected output type: {output.final_output_type}"
                    )
            (system.directory / "result.json").write_text(
                json.dumps(list(results.values()), indent=2) + "\n"
            )
        finally:
            model.close()

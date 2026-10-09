import json
from contextlib import closing

import soundfile
import torch
from transformers import AutoConfig, AutoProcessor

from stack.serving.system import System
from .media.inputs import prepare
from .omni_model import OmniModel


def run(package, args):
    if args.itrace or args.mtrace:
        raise ValueError("Omni system transport does not expose itrace/mtrace")
    torch.set_num_threads(args.threads)
    processor = AutoProcessor.from_pretrained(package.directory / "tokenizer", local_files_only=True)
    config = AutoConfig.from_pretrained(package.directory / "tokenizer", local_files_only=True)
    eos = processor.tokenizer.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    with System(args.simulator.resolve(), package.program, package.directory / "weights",
                args.log_dir.resolve(), args.memory_mib, args.timeout) as system:
        print(f"BEMU system: {system.directory}", flush=True)
        results = []
        with closing(OmniModel(system.directory, package.metadata, args.timeout, config)) as model:
            for index, item in enumerate(args.inputs):
                tokens, positions, delta, media = prepare(processor, item, args.run_config.resolve().parent,
                                                         config.thinker_config)
                generated, waveform = model.generate(tokens, positions, delta, media,
                                                      args.max_tokens, args.max_audio_tokens, args.temperature, eos)
                text = processor.tokenizer.decode(generated, skip_special_tokens=True)
                path = system.directory / f"{index}.wav"
                audio = waveform.detach().cpu().numpy().reshape(-1)
                soundfile.write(path, audio, 24000, subtype="FLOAT")
                result = {"request_id": str(index), "text": text, "token_ids": generated,
                          "audio": path.name, "samples": audio.size, "sample_rate": 24000}
                results.append(result)
                print(json.dumps(result), flush=True)
        (system.directory / "result.json").write_text(json.dumps(results, indent=2) + "\n")

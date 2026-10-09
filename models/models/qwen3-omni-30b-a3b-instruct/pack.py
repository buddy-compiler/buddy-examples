import argparse
import json
from pathlib import Path

from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from stack.compiler.package import pack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rax-pack", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.generated_dir.resolve(), args.model_dir.resolve()
    checkpoint = json.loads((source / "checkpoint.json").read_text())
    metadata = json.loads((source / "thinker/kernels.json").read_text())
    metadata["wave"] = json.loads((source / "wave/kernels.json").read_text())
    metadata["talker"] = json.loads((source / "talker/kernels.json").read_text())
    metadata["audio"] = json.loads((source / "audio/kernels.json").read_text())
    metadata["vision"] = json.loads((source / "vision/kernels.json").read_text())
    placements = json.loads((source / "weights/placements.json").read_text())
    placements.append(
        json.loads((source / "weights/vision-placement.json").read_text())
    )
    placements.append(json.loads((source / "weights/audio-placement.json").read_text()))
    placements.append(
        json.loads((source / "weights/talker-placement.json").read_text())
    )
    placements.append(json.loads((source / "weights/wave-placement.json").read_text()))
    for chip in metadata["system"]["chips"]:
        resident = sum(
            p["float_elements"] * 4 + p["weight_bytes"]
            for p in placements
            if p["chip"] == chip["id"]
        )
        if resident >= chip["memory_mib"] << 20:
            raise ValueError(f"chip {chip['id']} model weights exceed its DDR")
    metadata["placements"] = placements
    metadata.update(
        chip="omni", model=checkpoint["model"], revision=checkpoint["revision"]
    )
    (output / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    tokenizer = output / "tokenizer"
    tokens = AutoTokenizer.from_pretrained(checkpoint["path"])
    tokens.chat_template = AutoProcessor.from_pretrained(
        checkpoint["path"]
    ).chat_template
    tokens.save_pretrained(tokenizer)
    AutoProcessor.from_pretrained(checkpoint["path"]).save_pretrained(tokenizer)
    AutoConfig.from_pretrained(checkpoint["path"]).save_pretrained(tokenizer)
    resources = ["model.json"]
    for path in sorted((source / "weights").rglob("*")):
        if path.is_file():
            relative = path.relative_to(source)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.unlink(missing_ok=True)
            target.symlink_to(path)
            resources.append(str(relative))
    resources.extend(
        str(path.relative_to(output))
        for path in sorted(tokenizer.rglob("*"))
        if path.is_file()
    )
    pack(
        output,
        "omni",
        "qwen3-omni",
        checkpoint["model"],
        "omni-run",
        "model.json",
        resources,
        args.rax_pack,
        {
            "kind": "python",
            "entrypoint": "examples.models.omni.qwen3-omni-30b-a3b-instruct.serve:run",
        },
        "buckyball",
        embed_payload=False,
    )


if __name__ == "__main__":
    main()
